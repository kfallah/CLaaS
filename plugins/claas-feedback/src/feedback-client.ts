/**
 * HTTP client for the CLaaS feedback API.
 *
 * Uses native fetch() (Node 22+).
 * POST {claasApiUrl}/v1/feedback with the feedback payload.
 * 2-minute timeout to match CLaaS lock timeout.
 */

const TIMEOUT_MS = 2 * 60 * 1000; // 2 minutes

export interface FeedbackPayload {
  lora_id: string;
  prompt: string;
  response: string;
  feedback: string;
  training: {
    teacher_mode: string;
  };
}

export interface FeedbackResult {
  status: string;
  request_id?: string;
  lora_id?: string;
  timing_ms?: {
    total?: number;
    distill?: number;
    sleep?: number;
    wake?: number;
  };
}

export async function submitFeedback(
  claasApiUrl: string,
  payload: FeedbackPayload,
): Promise<FeedbackResult> {
  const url = `${claasApiUrl.replace(/\/+$/, "")}/v1/feedback`;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!res.ok) {
      const body = await res.text().catch(() => "");
      const truncated = body.slice(0, 500);

      if (res.status === 409) {
        throw new Error(
          "Another feedback update is in progress. Try again in a moment.",
        );
      }
      throw new Error(
        `CLaaS API returned ${res.status}: ${truncated}`,
      );
    }

    return (await res.json()) as FeedbackResult;
  } finally {
    clearTimeout(timer);
  }
}
