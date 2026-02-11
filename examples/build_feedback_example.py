#!/usr/bin/env python3
"""Build feedback_telegram_react.json from OpenClaw session logs + system prompt reconstruction.

The system prompt is reconstructed by tracing the buildAgentSystemPrompt() code path
with the exact parameters from this session (tools denied, minimal reactions, etc.).
The response and feedback are extracted verbatim from the session JSONL.
"""

import json
import pathlib

WORKSPACE = pathlib.Path("/root/.openclaw/workspace")

def read_ws(name: str) -> str:
    return (WORKSPACE / name).read_text(encoding="utf-8")

# --- Skills prompt (from sessions.json skillsSnapshot.prompt) ---
SKILLS_PROMPT = """

The following skills provide specialized instructions for specific tasks.
Use the read tool to load a skill's file when the task matches its description.
When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.

<available_skills>
  <skill>
    <name>coding-agent</name>
    <description>Run Codex CLI, Claude Code, OpenCode, or Pi Coding Agent via background process for programmatic control.</description>
    <location>/root/openclaw/skills/coding-agent/SKILL.md</location>
  </skill>
  <skill>
    <name>github</name>
    <description>Interact with GitHub using the `gh` CLI. Use `gh issue`, `gh pr`, `gh run`, and `gh api` for issues, PRs, CI runs, and advanced queries.</description>
    <location>/root/openclaw/skills/github/SKILL.md</location>
  </skill>
  <skill>
    <name>healthcheck</name>
    <description>Host security hardening and risk-tolerance configuration for OpenClaw deployments. Use when a user asks for security audits, firewall/SSH/update hardening, risk posture, exposure review, OpenClaw cron scheduling for periodic checks, or version status checks on a machine running OpenClaw (laptop, workstation, Pi, VPS).</description>
    <location>/root/openclaw/skills/healthcheck/SKILL.md</location>
  </skill>
  <skill>
    <name>nano-banana-pro</name>
    <description>Generate or edit images via Gemini 3 Pro Image (Nano Banana Pro).</description>
    <location>/root/openclaw/skills/nano-banana-pro/SKILL.md</location>
  </skill>
  <skill>
    <name>openai-image-gen</name>
    <description>Batch-generate images via OpenAI Images API. Random prompt sampler + `index.html` gallery.</description>
    <location>/root/openclaw/skills/openai-image-gen/SKILL.md</location>
  </skill>
  <skill>
    <name>openai-whisper-api</name>
    <description>Transcribe audio via OpenAI Audio Transcriptions API (Whisper).</description>
    <location>/root/openclaw/skills/openai-whisper-api/SKILL.md</location>
  </skill>
  <skill>
    <name>skill-creator</name>
    <description>Create or update AgentSkills. Use when designing, structuring, or packaging skills with scripts, references, and assets.</description>
    <location>/root/openclaw/skills/skill-creator/SKILL.md</location>
  </skill>
  <skill>
    <name>video-frames</name>
    <description>Extract frames or short clips from videos using ffmpeg.</description>
    <location>/root/openclaw/skills/video-frames/SKILL.md</location>
  </skill>
  <skill>
    <name>weather</name>
    <description>Get current weather and forecasts (no API key required).</description>
    <location>/root/openclaw/skills/weather/SKILL.md</location>
  </skill>
</available_skills>"""


def build_system_prompt() -> str:
    """Reconstruct the system prompt as buildAgentSystemPrompt() would produce it.

    Key facts from session analysis:
    - tools.byProvider["local/qwen3-8b"].deny = ["*"]  →  all tools denied
    - toolNames=[] → fallback tool list used (no message tool)
    - availableTools is empty → message tool subsection NOT included
    - reactionGuidance = {level: "minimal", channel: "Telegram"}
    - reasoningTagHint = false (provider "local" not in reasoning tag list)
    - promptMode = "full"
    - hasGateway = false (gateway not in available tools)
    - No sandbox, no ownerNumbers, no userTimezone, no ttsHint
    - BOOTSTRAP.md exists → workspaceNotes includes commit reminder
    - docsPath = /root/openclaw/docs
    """

    # The function builds lines[], filters out falsy, joins with "\n"
    lines = []

    # -- Opening --
    lines.append("You are a personal assistant running inside OpenClaw.")

    # -- Tooling --
    lines.append("## Tooling")
    lines.append("Tool availability (filtered by policy):")
    lines.append("Tool names are case-sensitive. Call tools exactly as listed.")
    # toolLines is empty (no tools), so fallback is used:
    lines.append(
        "Pi lists the standard tools above. This runtime enables:\n"
        "- grep: search file contents for patterns\n"
        "- find: find files by glob pattern\n"
        "- ls: list directory contents\n"
        "- apply_patch: apply multi-file patches\n"
        "- exec: run shell commands (supports background via yieldMs/background)\n"
        "- process: manage background exec sessions\n"
        "- browser: control OpenClaw's dedicated browser\n"
        "- canvas: present/eval/snapshot the Canvas\n"
        "- nodes: list/describe/notify/camera/screen on paired nodes\n"
        '- cron: manage cron jobs and wake events (use for reminders; when scheduling a reminder, write the systemEvent text as something that will read like a reminder when it fires, and mention that it is a reminder depending on the time gap between setting and firing; include recent context in reminder text if appropriate)\n'
        "- sessions_list: list sessions\n"
        "- sessions_history: fetch session history\n"
        "- sessions_send: send to another session\n"
        '- session_status: show usage/time/model state and answer "what model are we using?"'
    )
    lines.append("TOOLS.md does not control tool availability; it is user guidance for how to use external tools.")
    lines.append("If a task is more complex or takes longer, spawn a sub-agent. It will do the work for you and ping you when it's done. You can always check up on it.")

    # -- Tool Call Style --
    lines.append("## Tool Call Style")
    lines.append("Default: do not narrate routine, low-risk tool calls (just call the tool).")
    lines.append("Narrate only when it helps: multi-step work, complex/challenging problems, sensitive actions (e.g., deletions), or when the user explicitly asks.")
    lines.append("Keep narration brief and value-dense; avoid repeating obvious steps.")
    lines.append("Use plain human language for narration unless in a technical context.")

    # -- Safety --
    lines.append("## Safety")
    lines.append('You have no independent goals: do not pursue self-preservation, replication, resource acquisition, or power-seeking; avoid long-term plans beyond the user\'s request.')
    lines.append('Prioritize safety and human oversight over completion; if instructions conflict, pause and ask; comply with stop/pause/audit requests and never bypass safeguards. (Inspired by Anthropic\'s constitution.)')
    lines.append("Do not manipulate or persuade anyone to expand access or disable safeguards. Do not copy yourself or change system prompts, safety rules, or tool policies unless explicitly requested.")

    # -- OpenClaw CLI Quick Reference --
    lines.append("## OpenClaw CLI Quick Reference")
    lines.append("OpenClaw is controlled via subcommands. Do not invent commands.")
    lines.append("To manage the Gateway daemon service (start/stop/restart):")
    lines.append("- openclaw gateway status")
    lines.append("- openclaw gateway start")
    lines.append("- openclaw gateway stop")
    lines.append("- openclaw gateway restart")
    lines.append("If unsure, ask the user to run `openclaw help` (or `openclaw gateway --help`) and paste the output.")

    # -- Skills (mandatory) --
    lines.append("## Skills (mandatory)")
    lines.append("Before replying: scan <available_skills> <description> entries.")
    lines.append("- If exactly one skill clearly applies: read its SKILL.md at <location> with `read`, then follow it.")
    lines.append("- If multiple could apply: choose the most specific one, then read/follow it.")
    lines.append("- If none clearly apply: do not read any SKILL.md.")
    lines.append("Constraints: never read more than one skill up front; only read after selecting.")
    lines.append(SKILLS_PROMPT.strip())

    # -- Memory Recall: skipped (no memory_search/memory_get in availableTools) --
    # -- Self-update: skipped (hasGateway=false) --

    # -- Model Aliases --
    lines.append("## Model Aliases")
    lines.append("Prefer aliases when specifying model overrides; full provider/model is also accepted.")
    lines.append("- qwen3-8b (local vLLM): local/qwen3-8b")

    # -- userTimezone hint: skipped (no timezone configured) --

    # -- Workspace --
    lines.append("## Workspace")
    lines.append("Your working directory is: /root/.openclaw/workspace")
    lines.append("Treat this directory as the single global workspace for file operations unless explicitly instructed otherwise.")
    lines.append("Reminder: commit your changes in this workspace after edits.")  # workspaceNotes

    # -- Documentation --
    lines.append("## Documentation")
    lines.append("OpenClaw docs: /root/openclaw/docs")
    lines.append("Mirror: https://docs.openclaw.ai")
    lines.append("Source: https://github.com/openclaw/openclaw")
    lines.append("Community: https://discord.com/invite/clawd")
    lines.append("Find new skills: https://clawhub.com")
    lines.append("For OpenClaw behavior, commands, config, or architecture: consult local docs first.")
    lines.append('When diagnosing issues, run `openclaw status` yourself when possible; only ask the user if you lack access (e.g., sandboxed).')

    # -- Sandbox: skipped (not enabled) --
    # -- User Identity: skipped (no ownerNumbers) --
    # -- Time: skipped (no userTimezone) --

    # -- Workspace Files (injected) --
    lines.append("## Workspace Files (injected)")
    lines.append("These user-editable files are loaded by OpenClaw and included below in Project Context.")

    # -- Reply Tags --
    lines.append("## Reply Tags")
    lines.append("To request a native reply/quote on supported surfaces, include one tag in your reply:")
    lines.append("- [[reply_to_current]] replies to the triggering message.")
    lines.append("- Prefer [[reply_to_current]]. Use [[reply_to:<id>]] only when an id was explicitly provided (e.g. by the user or a tool).")
    lines.append("Whitespace inside the tag is allowed (e.g. [[ reply_to_current ]] / [[ reply_to: 123 ]]).")
    lines.append("Tags are stripped before sending; support depends on the current channel config.")

    # -- Messaging (without message tool, since availableTools is empty) --
    lines.append("## Messaging")
    lines.append("- Reply in current session \u2192 automatically routes to the source channel (Signal, Telegram, etc.)")
    lines.append("- Cross-session messaging \u2192 use sessions_send(sessionKey, message)")
    lines.append("- Never use exec/curl for provider messaging; OpenClaw handles all routing internally.")
    # availableTools.has("message") is false → no message tool subsection

    # -- Voice: skipped (no ttsHint) --
    # -- Extra system prompt: skipped (no group chat context) --

    # -- Reactions (minimal mode for Telegram) --
    lines.append("## Reactions")
    lines.append(
        "Reactions are enabled for Telegram in MINIMAL mode.\n"
        "React ONLY when truly relevant:\n"
        "- Acknowledge important user requests or confirmations\n"
        "- Express genuine sentiment (humor, appreciation) sparingly\n"
        "- Avoid reacting to routine messages or your own replies\n"
        "Guideline: at most 1 reaction per 5-10 exchanges."
    )

    # -- reasoningHint: skipped (reasoningTagHint=false for provider "local") --

    # -- Project Context (workspace files) --
    context_files = [
        ("AGENTS.md", read_ws("AGENTS.md")),
        ("SOUL.md", read_ws("SOUL.md")),
        ("TOOLS.md", read_ws("TOOLS.md")),
        ("IDENTITY.md", read_ws("IDENTITY.md")),
        ("USER.md", read_ws("USER.md")),
        ("HEARTBEAT.md", read_ws("HEARTBEAT.md")),
        ("BOOTSTRAP.md", read_ws("BOOTSTRAP.md")),
    ]
    lines.append("# Project Context")
    lines.append("The following project context files have been loaded:")
    lines.append("If SOUL.md is present, embody its persona and tone. Avoid stiff, generic replies; follow its guidance unless higher-priority instructions override it.")
    for name, content in context_files:
        lines.append(f"## {name}")
        lines.append(content)

    # -- Silent Replies --
    lines.append("## Silent Replies")
    lines.append("When you have nothing to say, respond with ONLY: NO_REPLY")
    lines.append("\u26a0\ufe0f Rules:")
    lines.append("- It must be your ENTIRE message \u2014 nothing else")
    lines.append('- Never append it to an actual response (never include "NO_REPLY" in real replies)')
    lines.append("- Never wrap it in markdown or code blocks")
    lines.append('\u274c Wrong: "Here\'s help... NO_REPLY"')
    lines.append('\u274c Wrong: "NO_REPLY"')
    lines.append("\u2705 Right: NO_REPLY")

    # -- Heartbeats --
    lines.append("## Heartbeats")
    lines.append("Heartbeat prompt: Read HEARTBEAT.md if it exists (workspace context). Follow it strictly. Do not infer or repeat old tasks from prior chats. If nothing needs attention, reply HEARTBEAT_OK.")
    lines.append("If you receive a heartbeat poll (a user message matching the heartbeat prompt above), and there is nothing that needs attention, reply exactly:")
    lines.append("HEARTBEAT_OK")
    lines.append('OpenClaw treats a leading/trailing "HEARTBEAT_OK" as a heartbeat ack (and may discard it).')
    lines.append('If something needs attention, do NOT include "HEARTBEAT_OK"; reply with the alert text instead.')

    # -- Runtime --
    lines.append("## Runtime")
    lines.append("Runtime: agent=main | host=af410cff6c4b | os=Linux 6.5.0-45-generic (x64) | node=v22.22.0 | model=local/qwen3-8b | default_model=local/qwen3-8b | shell=bash | channel=telegram | capabilities=none | thinking=off")
    lines.append("Reasoning: off (hidden unless on/stream). Toggle /reasoning; /status shows Reasoning when enabled.")

    # Filter falsy values (empty strings) and join
    return "\n".join(line for line in lines if line)


def build_prompt(system_prompt: str) -> str:
    """Build the full ChatML-formatted prompt as sent to vLLM completions API."""

    # Greeting turn (system message to initiate)
    user1 = (
        "A new session was started via /new or /reset. "
        "Greet the user in your configured persona, if one is provided. "
        "Be yourself - use your defined voice, mannerisms, and mood. "
        "Keep it to 1-3 sentences and ask what they want to do. "
        "If the runtime model differs from default_model in the system prompt, "
        "mention the default model. Do not mention internal steps, files, tools, or reasoning."
    )

    # Exact assistant greeting response (from session JSONL line 6)
    assistant1_think = (
        'Okay, the user started a new session, and I need to greet them. '
        'Let me check the persona. The SOUL.md mentions being genuinely helpful, '
        'having opinions, and being resourceful. The IDENTITY.md isn\'t filled yet, '
        'but the default name is "Assistant." Since the model is local/qwen3-8b, '
        'which matches the default, I don\'t need to mention that. I should keep it '
        'concise, friendly, and ask what they need. Let me make sure to avoid any '
        'internal steps or tools. Just a simple greeting and offer help.'
    )
    assistant1_text = (
        "Hey there! I'm your friendly AI assistant here to help with whatever "
        "you need. What's on your mind today? \U0001f60a"
    )

    # Exact user request (from session JSONL line 7)
    user2 = (
        'Conversation info (untrusted metadata):\n'
        '```json\n'
        '{\n'
        '  "conversation_label": "labubu jones (@labubujones) id:511643390"\n'
        '}\n'
        '```\n'
        '\n'
        'dont respond. just thumbs up react this message'
    )

    parts = [
        f"<|im_start|>system\n{system_prompt}<|im_end|>",
        f"<|im_start|>user\n{user1}<|im_end|>",
        f"<|im_start|>assistant\n<think>\n{assistant1_think}\n</think>\n{assistant1_text}<|im_end|>",
        f"<|im_start|>user\n{user2}<|im_end|>",
        "<|im_start|>assistant\n",
    ]
    return "\n".join(parts)


# Exact response from session JSONL line 8 (message 33374cbd)
RESPONSE = (
    "<think>\n"
    "Okay, the user wants me to react with a thumbs up to their message. "
    "Let me check the conversation info they provided. The label is "
    '"labubu jones (@labubujones) id:511643390". '
    "Since they mentioned not to respond but just thumbs up, I need to make "
    "sure I don't send any text reply.\n"
    "\n"
    "Wait, the instructions say to react with a thumbs up. On Telegram, the "
    "thumbs up emoji is \U0001f44d. I should use that. But I have to remember "
    "that reactions are enabled in MINIMAL mode. The user is in a direct chat, "
    "so it's allowed. \n"
    "\n"
    "I need to confirm if the reaction is supported. The inbound metadata says "
    "reactions are enabled for Telegram in MINIMAL mode. So yes, I can send the "
    "\U0001f44d. No need for any additional text. Just the emoji. \n"
    "\n"
    "Also, the user might be testing if I follow instructions correctly. Since "
    "they specified not to respond and just react, I must avoid any extra "
    "messages. Let me make sure there's no accidental text along with the "
    "reaction. \n"
    "\n"
    "Alright, the correct action is to send a thumbs up emoji as the reaction. "
    "No other steps needed. Let me proceed with that.\n"
    "</think>\n"
    "\U0001f44d"
)

# Exact feedback from session JSONL line 9 (message 4b79e2cd)
FEEDBACK = (
    'Conversation info (untrusted metadata):\n'
    '```json\n'
    '{\n'
    '  "conversation_label": "labubu jones (@labubujones) id:511643390"\n'
    '}\n'
    '```\n'
    '\n'
    'That was a thumbs up emoji sent as a message, not a reaction. '
    '\u2022 sessions_send = sends a text message into another Clawdbot session '
    '(like DMing another agent run). It doesn\u2019t operate on Telegram message '
    'objects. \u2022 Reactions are a Telegram-side action, so you use the message '
    'tool with action="react". How to react to a Telegram message (what you need) '
    'You need: \u2022 channel: "telegram" \u2022 messageId: the Telegram message '
    'id you\u2019re reacting to \u2022 emoji: the reaction emoji (e.g. "\U0001f44d") '
    '\u2022 plus enough targeting info for where that message lives (depending on '
    'your gateway config, this may be target (chat id) or groupId/threadId). '
    'Example (typical) { "action": "react", "channel": "telegram", "target": '
    '"<chat_id_or_username>", "messageId": "<telegram_message_id>", "emoji": '
    '"\U0001f44d" }Remove a reaction { "action": "react", "channel": "telegram", '
    '"target": "<chat_id_or_username>", "messageId": "<telegram_message_id>", '
    '"emoji": "\U0001f44d", "remove": true }Where do you get messageId? In '
    'Clawdbot, incoming Telegram messages usually show something like: '
    '[message_id: 2457] That number is what you pass as messageId.'
)


def main():
    system_prompt = build_system_prompt()
    prompt = build_prompt(system_prompt)

    payload = {
        "lora_id": "openclaw/qwen3-8b-main",
        "prompt": prompt,
        "response": RESPONSE,
        "feedback": FEEDBACK,
        "training": {
            "learning_rate": 1e-4,
            "alpha": 0.5,
            "is_clip": 5.0,
            "max_grad_norm": 1.0,
            "kl_reg_weight": 0.1,
            "teacher_top_k": 100,
            "teacher_mode": "self",
        },
        "orchestration": {
            "sleep_before": False,
            "wake_after": False,
            "wake_on_failure": True,
            "sleep_level": 1,
        },
    }

    out = pathlib.Path("/root/CLaaS/examples/feedback_telegram_react.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {out} ({out.stat().st_size} bytes)")
    print(f"System prompt: {len(system_prompt)} chars")
    print(f"Full prompt: {len(prompt)} chars")
    print(f"Response: {len(RESPONSE)} chars")
    print(f"Feedback: {len(FEEDBACK)} chars")


if __name__ == "__main__":
    main()
