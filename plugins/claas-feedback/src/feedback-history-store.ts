/**
 * In-memory feedback history and pending batch buffers.
 */

export interface FeedbackHistoryEntry {
  content_hash: string;
  feedback: string;
  user_prompt: string;
  timestamp: number;
}

const historyBySession = new Map<string, FeedbackHistoryEntry[]>();
const pendingBySession = new Map<string, FeedbackHistoryEntry[]>();
const MAX_HISTORY_PER_SESSION = 200;
const MAX_PENDING_PER_SESSION = 200;

/**
 * Append one feedback item and return pending-buffer stats.
 */
export function appendFeedback(
  sessionKey: string,
  contentHash: string,
  feedback: string,
  userPrompt: string,
): { pendingSize: number } {
  const entry: FeedbackHistoryEntry = {
    content_hash: contentHash,
    feedback,
    user_prompt: userPrompt,
    timestamp: Date.now(),
  };

  const history = historyBySession.get(sessionKey) ?? [];
  history.push(entry);
  if (history.length > MAX_HISTORY_PER_SESSION) {
    history.splice(0, history.length - MAX_HISTORY_PER_SESSION);
  }
  historyBySession.set(sessionKey, history);

  const pending = pendingBySession.get(sessionKey) ?? [];
  pending.push(entry);
  if (pending.length > MAX_PENDING_PER_SESSION) {
    pending.splice(0, pending.length - MAX_PENDING_PER_SESSION);
  }
  pendingBySession.set(sessionKey, pending);

  return { pendingSize: pending.length };
}

/**
 * Remove and return the oldest pending entries for a session.
 */
export function takePendingBatch(sessionKey: string, batchSize: number): FeedbackHistoryEntry[] {
  if (batchSize <= 0) {
    return [];
  }
  const pending = pendingBySession.get(sessionKey) ?? [];
  const batch = pending.slice(0, batchSize);
  pendingBySession.set(sessionKey, pending.slice(batchSize));
  return batch;
}


/**
 * Prepend a previously dequeued batch back to pending entries for retry.
 */
export function requeuePendingBatch(sessionKey: string, batch: FeedbackHistoryEntry[]): void {
  if (batch.length === 0) {
    return;
  }
  const pending = pendingBySession.get(sessionKey) ?? [];
  const merged = [...batch, ...pending];
  pendingBySession.set(sessionKey, merged.slice(0, MAX_PENDING_PER_SESSION));
}

/**
 * Return the current pending-buffer size for a session.
 */
export function getPendingSize(sessionKey: string): number {
  const pending = pendingBySession.get(sessionKey);
  if (!pending) {
    return 0;
  }
  return pending.length;
}
