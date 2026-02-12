/**
 * In-memory store for per-sender conversation context.
 *
 * Key:   "channel:senderId"  (e.g. "telegram:511643390")
 * Value: { messages, sessionKey, timestamp }
 *
 * TTL: 1 hour — stale entries are evicted on access and periodically.
 * Max entries: 1000 — oldest entries are evicted when the limit is reached.
 */

export interface ConversationContext {
  messages: unknown[];
  sessionKey: string;
  timestamp: number;
}

const TTL_MS = 60 * 60 * 1000; // 1 hour
const MAX_ENTRIES = 1000;

const store = new Map<string, ConversationContext>();

function isStale(entry: ConversationContext): boolean {
  return Date.now() - entry.timestamp > TTL_MS;
}

/** Remove all entries older than TTL. */
function evictStale(): void {
  for (const [key, value] of store) {
    if (isStale(value)) {
      store.delete(key);
    }
  }
}

/** If over MAX_ENTRIES, drop the oldest entries until at limit. */
function evictOldest(): void {
  if (store.size <= MAX_ENTRIES) return;

  const sorted = [...store.entries()].sort(
    (a, b) => a[1].timestamp - b[1].timestamp,
  );
  const toRemove = sorted.slice(0, store.size - MAX_ENTRIES);
  for (const [key] of toRemove) {
    store.delete(key);
  }
}

export function set(key: string, ctx: ConversationContext): void {
  evictStale();
  store.set(key, ctx);
  evictOldest();
}

export function get(key: string): ConversationContext | undefined {
  const entry = store.get(key);
  if (!entry) return undefined;
  if (isStale(entry)) {
    store.delete(key);
    return undefined;
  }
  return entry;
}
