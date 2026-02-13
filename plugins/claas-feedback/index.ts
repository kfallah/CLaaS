/**
 * CLaaS Feedback Plugin for OpenClaw
 *
 * Registers:
 * - message_received hook: pushes sender key to a queue for agent_end to consume
 * - agent_end hook: pops sender key, caches { messages } keyed by sender
 * - /feedback command: looks up cached context, builds ChatML, calls CLaaS API
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

import * as contextStore from "./src/context-store.ts";
import { buildChatML } from "./src/chatml.ts";
import { submitFeedback } from "./src/feedback-client.ts";

// FIFO queue: message_received pushes sender keys, agent_end pops them.
const pendingSenders: string[] = [];

export default function register(api: OpenClawPluginApi) {
  const config = api.config ?? ({} as Record<string, unknown>);
  const claasApiUrl: string = (config as any).claasApiUrl ?? "http://localhost:8080";
  const loraId: string = (config as any).loraId ?? "openclaw/assistant-latest";

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
    console.log(`[claas-feedback] message_received: channelId=${channelId} from=${from} rawFrom=${rawFrom} senderKey=${senderKey}`);
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

    console.log(`[claas-feedback] agent_end: senderKey=${senderKey} messageCount=${messages.length}`);
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
      console.log(`[claas-feedback] /feedback lookup: senderKey=${senderKey} from=${from}`);
      console.log(`[claas-feedback] contextStore keys: ${JSON.stringify(contextStore.keys())}`);
      const cached = contextStore.get(senderKey);
      if (!cached) {
        return {
          text: "No recent conversation found. Chat with the bot first, then use /feedback.",
        };
      }

      // Build ChatML prompt/response from cached messages
      const chatML = buildChatML(cached.messages);
      if (!chatML) {
        return { text: "No bot response found in the last conversation." };
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
      const startMs = Date.now();
      try {
        const result = await submitFeedback(claasApiUrl, {
          lora_id: loraId,
          prompt: chatML.prompt,
          response: chatML.response,
          feedback: feedbackText,
          training: { teacher_mode: "self" },
        });

        const elapsed = Date.now() - startMs;
        const totalMs = result.timing_ms?.total ?? elapsed;
        const seconds = (totalMs / 1000).toFixed(1);
        const meta = result.distill_result?.metadata;
        const loss = meta?.total_loss;
        const tokens = meta?.tokens_processed;

        let detail = `${seconds}s`;
        if (loss !== undefined) detail += ` | loss: ${loss.toFixed(4)}`;
        if (tokens !== undefined) detail += ` | ${tokens} tokens`;

        return {
          text: `\u2705 Feedback applied (${detail})`,
        };
      } catch (err: any) {
        return { text: `\u274C Feedback failed: ${err.message}` };
      }
    },
  });
}
