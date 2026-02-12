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

    if (!senderKey || !Array.isArray(messages) || messages.length === 0) return;

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

      // Look up cached conversation for this sender
      const channel = ctx.channel ?? ctx.channelId;
      const senderId = ctx.senderId ?? ctx.from;
      if (!channel || !senderId) {
        return { text: "Could not identify sender. Please try again." };
      }

      const senderKey = `${channel}:${senderId}`;
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

        return {
          text: `Feedback submitted successfully (${seconds}s). The model will improve from your input.`,
        };
      } catch (err: any) {
        return { text: `Feedback failed: ${err.message}` };
      }
    },
  });
}
