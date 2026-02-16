/**
 * HTTP client for the CLaaS batched feedback API.
 */

const TIMEOUT_MS = 10 * 60 * 1000;

export interface DistillRequestPayload {
  lora_id: string;
  prompt: string;
  response: string;
  feedback: string;
  rollout_logprobs?: number[] | null;
  training: {
    teacher_mode: string;
  };
}

export interface FeedbackBatchPayload {
  requests: DistillRequestPayload[];
  orchestration: {
    sleep_before: boolean;
    wake_after: boolean;
    wake_on_failure: boolean;
    sleep_level: number;
  };
}

export interface FeedbackResult {
  status: string;
  request_id?: string;
  lora_id?: string;
  batch_size?: number;
  timing_ms?: {
    total?: number;
    distill?: number;
    sleep?: number;
    wake?: number;
  };
  distill_result?: {
    metadata?: {
      total_loss?: number;
      distill_loss?: number;
      tokens_processed?: number;
      grad_norm?: number;
      batch_size?: number;
    };
  };
}

/**
 * Submit one explicit batch to CLaaS.
 */
export async function submitFeedback(
  claasApiUrl: string,
  payload: FeedbackBatchPayload,
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
      const body = await res.text();
      throw new Error(`CLaaS API returned ${res.status}: ${body.slice(0, 500)}`);
    }

    return (await res.json()) as FeedbackResult;
  } catch (err: unknown) {
    if (err instanceof Error && err.name === "AbortError") {
      throw new Error(`CLaaS API request timed out after ${Math.round(TIMEOUT_MS / 1000)}s`);
    }
    if (err instanceof Error) {
      throw err;
    }
    throw new Error(String(err));
  } finally {
    clearTimeout(timer);
  }
}
