/**
 * CLaaS Feedback Plugin for OpenClaw
 *
 * Registers:
 * - message_received hook: pushes sender key to a queue for agent_end to consume
 * - agent_end hook: pops sender key, caches { messages } keyed by sender
 * - /feedback command: looks up cached context, fetches raw completion from proxy, calls CLaaS API
 *
 * message_received fires synchronously before the agent processes a message,
 * and agent_end fires after, so a simple FIFO queue links the two.
 */

import type {
  OpenClawPluginApi,
  PluginHookMessageReceivedEvent,
  PluginHookMessageContext,
  PluginHookAgentEndEvent,
  PluginHookAgentContext,
  PluginCommandContext,
} from "openclaw/plugin-sdk";

import { createHash } from "node:crypto";

import * as contextStore from "./src/context-store.ts";
import { extractContent, fetchRawCompletion } from "./src/chatml.ts";
import { submitFeedback } from "./src/feedback-client.ts";
import { appendFeedback, getPendingSize, takePendingBatch, requeuePendingBatch } from "./src/feedback-history-store.ts";

// FIFO queue: message_received pushes sender keys, agent_end pops them.
const pendingSenders: string[] = [];

const redactIdentifier = (value: string): string => {
  const firstColon = value.indexOf(":");
  if (firstColon < 0) return "***";
  return `${value.slice(0, firstColon + 1)}***`;
};

interface ClaasConfig {
  claasApiUrl?: string;
  proxyUrl?: string;
  loraId?: string;
  debug?: boolean;
  feedbackBatchSize?: number;
}

export default function register(api: OpenClawPluginApi) {
  const config = (api.config ?? {}) as ClaasConfig;
  const claasApiUrl =
    (typeof config.claasApiUrl === "string" && config.claasApiUrl.trim()) ||
    (typeof process.env.CLAAS_API_URL === "string" && process.env.CLAAS_API_URL.trim()) ||
    "http://claas-api:8080";
  const proxyUrl =
    (typeof config.proxyUrl === "string" && config.proxyUrl.trim()) ||
    (typeof process.env.CLAAS_TINKER_PROXY_URL === "string" && process.env.CLAAS_TINKER_PROXY_URL.trim()) ||
    (typeof process.env.CLAAS_VLLM_BASE_URL === "string" && process.env.CLAAS_VLLM_BASE_URL.trim()) ||
    "http://tinker-proxy:8000";
  const loraId = config.loraId ?? "openclaw/assistant-latest";
  const debugEnabled = config.debug === true || process.env.CLAAS_FEEDBACK_DEBUG === "true";
  const feedbackBatchSize = Math.max(1, config.feedbackBatchSize ?? 4);
  const logDebug = (message: string): void => {
    if (debugEnabled) {
      console.debug(message);
    }
  };

  // -----------------------------------------------------------------------
  // Hook: message_received
  // Fires when a message arrives, before the agent processes it.
  // Push the sender key so agent_end can associate messages with this sender.
  // -----------------------------------------------------------------------
  api.on("message_received", (event: PluginHookMessageReceivedEvent, ctx: PluginHookMessageContext) => {
    const channelId = ctx.channelId;
    const from = event.from;
    if (!channelId || !from) return;

    // event.from may already be prefixed (e.g. "telegram:511643390"), so strip
    // the channel prefix if present to get the raw sender ID, then build a
    // canonical key that matches what the /feedback command constructs.
    const prefix = `${channelId}:`;
    const rawFrom = from.startsWith(prefix) ? from.slice(prefix.length) : from;
    const senderKey = `${channelId}:${rawFrom}`;
    logDebug(
      `[claas-feedback] message_received: channelId=${channelId} senderKey=${redactIdentifier(senderKey)}`,
    );
    pendingSenders.push(senderKey);

    // Safety: cap queue at 50 to prevent unbounded growth
    while (pendingSenders.length > 50) {
      pendingSenders.shift();
    }
  });

  // -----------------------------------------------------------------------
  // Hook: agent_end
  // Fires after the agent finishes processing. Pop the sender key from the
  // queue and cache the full message history for /feedback to use.
  // -----------------------------------------------------------------------
  api.on("agent_end", (event: PluginHookAgentEndEvent, _ctx: PluginHookAgentContext) => {
    const messages = event.messages;
    const senderKey = pendingSenders.shift();

    if (!senderKey) {
      console.warn("[claas-feedback] agent_end: missing sender key; skipping context cache for this run");
      return;
    }
    if (!Array.isArray(messages) || messages.length === 0) return;

    logDebug(
      `[claas-feedback] agent_end: senderKey=${redactIdentifier(senderKey)} messageCount=${messages.length}`,
    );
    contextStore.set(senderKey, {
      messages,
      sessionKey: senderKey,
      timestamp: Date.now(),
    });
  });

  // -----------------------------------------------------------------------
  // Command: /feedback <text>
  // -----------------------------------------------------------------------
  api.registerCommand({
    name: "feedback",
    description: "Submit feedback on the last bot response to improve the model",
    acceptsArgs: true,
    handler: async (ctx: PluginCommandContext) => {
      const feedbackText = (ctx.args ?? "").trim();
      if (!feedbackText) {
        return { text: "Usage: /feedback <your feedback here>" };
      }

      // Look up cached conversation for this chat.
      // message_received builds its key by stripping a redundant channel prefix
      // from event.from:
      //   DM:    event.from="telegram:511643390"        → key "telegram:511643390"
      //   Group: event.from="telegram:group:-5242423293" → key "telegram:group:-5242423293"
      //
      // In the command handler ctx.from carries the same value as event.from,
      // so we replicate the exact same key-building logic here.
      const channel = ctx.channel ?? ctx.channelId;
      const from = ctx.from;
      if (!channel || !from) {
        return { text: "Could not identify sender. Please try again." };
      }

      const prefix = `${channel}:`;
      const rawFrom = from.startsWith(prefix) ? from.slice(prefix.length) : from;
      const senderKey = `${channel}:${rawFrom}`;
      logDebug(`[claas-feedback] /feedback lookup: senderKey=${redactIdentifier(senderKey)}`);
      const cached = contextStore.get(senderKey);
      if (!cached) {
        return {
          text: "No recent conversation found. Chat with the bot first, then use /feedback.",
        };
      }

      // Extract the last assistant content and look up the raw completion
      // from the inference proxy cache (keyed by SHA-256 of parsed content).
      const lastAssistant = cached.messages
        .slice()
        .reverse()
        .find((m: Record<string, unknown>) => m.role === "assistant");
      if (!lastAssistant) {
        return { text: "No bot response found in the last conversation." };
      }
      const parsedContent = extractContent((lastAssistant as Record<string, unknown>).content);
      if (!parsedContent) {
        return { text: "No bot response found in the last conversation." };
      }
      const contentHash = createHash("sha256").update(parsedContent).digest("hex");

      let rawPrompt: string;
      let rawResponse: string;
      let rolloutLogprobs: number[] | null = null;
      try {
        const raw = await fetchRawCompletion(proxyUrl, contentHash);
        rawPrompt = raw.prompt;
        rawResponse = raw.response;
        rolloutLogprobs = raw.logprobs;
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(`[claas-feedback] raw completion fetch failed: ${msg}`);
        return {
          text: `\u274C Raw completion not available. The proxy may have restarted or the completion expired.`,
        };
      }

      // Send a "processing" indicator before the long-running CLaaS call
      const replyTo = ctx.to ?? ctx.senderId;
      const sendOpts: Record<string, unknown> = {};
      if (ctx.messageThreadId) sendOpts.messageThreadId = ctx.messageThreadId;
      if (ctx.accountId) sendOpts.accountId = ctx.accountId;

      const sendTelegram = api.runtime?.channel?.telegram?.sendMessageTelegram;
      if (sendTelegram && replyTo) {
        try {
          await sendTelegram(replyTo, "\u{1F9E0} Learning from your feedback\u2026", sendOpts);
        } catch {
          // Best-effort; don't block on notification failure
        }
      }

      // Submit to CLaaS
      const { pendingSize } = appendFeedback(senderKey, rawPrompt, rawResponse, feedbackText, rolloutLogprobs);
      if (pendingSize < feedbackBatchSize) {
        return {
          text: `✅ Feedback queued (buffer: ${pendingSize}/${feedbackBatchSize})`,
        };
      }

      const batch = takePendingBatch(senderKey, feedbackBatchSize);
      if (batch.length !== feedbackBatchSize) {
        throw new Error("pending feedback batch size mismatch");
      }

      const startMs = Date.now();
      try {
        const result = await submitFeedback(claasApiUrl, {
          requests: batch.map((item) => ({
            lora_id: loraId,
            prompt: item.prompt,
            response: item.response,
            feedback: item.feedback,
            rollout_logprobs: item.rollout_logprobs,
            training: { teacher_mode: "self" },
          })),
          orchestration: {
            sleep_before: true,
            wake_after: true,
            wake_on_failure: true,
            sleep_level: 1,
          },
        });

        const elapsed = Date.now() - startMs;
        const totalMs = result.timing_ms?.total ?? elapsed;
        const seconds = (totalMs / 1000).toFixed(1);
        const meta = result.distill_result?.metadata;
        const loss = meta?.total_loss;
        const tokens = meta?.tokens_processed;
        const remaining = getPendingSize(senderKey);

        let detail = `${seconds}s`;
        if (loss !== undefined) detail += ` | loss: ${loss.toFixed(4)}`;
        if (tokens !== undefined) detail += ` | ${tokens} tokens`;
        detail += ` | buffer: ${remaining}/${feedbackBatchSize}`;

        return {
          text: `✅ Feedback applied (${detail})`,
        };
      } catch (err: unknown) {
        requeuePendingBatch(senderKey, batch);
        const msg = err instanceof Error ? err.message : String(err);
        console.error(
          `[claas-feedback] /feedback failed: api=${claasApiUrl} lora=${loraId} error=${msg}`,
        );
        return { text: `\u274C Feedback failed (batch re-queued): ${msg}` };
      }
    },
  });
}
