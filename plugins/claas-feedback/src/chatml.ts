/**
 * Reconstruct ChatML prompt/response pair from an AgentMessage[] array.
 *
 * Iterates messages, keeps only user/assistant roles, and formats each as:
 *   <|im_start|>role\ncontent<|im_end|>
 *
 * Content can be a plain string or a ContentBlock[] array (with `thinking`
 * blocks wrapped in <think> tags).
 *
 * Splits at the last assistant message:
 *   - Everything before → `prompt`
 *   - Last assistant content → `response`
 */

const IM_START = "<|im_start|>";
const IM_END = "<|im_end|>";

interface ContentBlock {
  type: string;
  text?: string;
  thinking?: string;
  content?: string;
}

/** Extract text from a content field that may be string or ContentBlock[]. */
export function extractContent(content: unknown): string {
  if (typeof content === "string") return content;

  if (!Array.isArray(content)) return String(content ?? "");

  const parts: string[] = [];
  for (const block of content as ContentBlock[]) {
    if (block.type === "thinking" && (block.thinking || block.text)) {
      parts.push(`<think>${block.thinking || block.text}</think>`);
    } else if (block.type === "text" && block.text) {
      parts.push(block.text);
    } else if (block.text) {
      parts.push(block.text);
    } else if (block.content) {
      parts.push(String(block.content));
    }
  }
  return parts.join("\n");
}

function formatTurn(role: string, content: string): string {
  return `${IM_START}${role}\n${content}${IM_END}`;
}

interface ChatMLResult {
  prompt: string;
  response: string;
}

/**
 * Build ChatML prompt and response from an array of agent messages.
 * Returns null if no assistant message is found.
 */
export function buildChatML(
  messages: unknown[],
): ChatMLResult | null {
  // Filter to user/assistant turns
  const turns: { role: string; content: string }[] = [];
  for (const msg of messages) {
    const m = msg as Record<string, unknown>;
    const role = String(m.role ?? "");
    if (role !== "user" && role !== "assistant") continue;
    const content = extractContent(m.content);
    if (!content) continue;
    turns.push({ role, content });
  }

  // Find last assistant message
  let lastAssistantIdx = -1;
  for (let i = turns.length - 1; i >= 0; i--) {
    if (turns[i].role === "assistant") {
      lastAssistantIdx = i;
      break;
    }
  }

  if (lastAssistantIdx === -1) return null;

  // Everything before the last assistant → prompt
  const promptParts = turns
    .slice(0, lastAssistantIdx)
    .map((t) => formatTurn(t.role, t.content));
  const prompt = promptParts.join("\n");

  // Last assistant content → response
  const response = turns[lastAssistantIdx].content;

  return { prompt, response };
}

