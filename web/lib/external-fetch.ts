const DEFAULT_TIMEOUT_MS = 20_000;
const DEFAULT_RETRIES = 2;
const DEFAULT_RETRY_DELAY_MS = 350;

const DEFAULT_RETRYABLE_STATUS_CODES = new Set([408, 425, 429, 500, 502, 503, 504]);

type ExternalFetchOptions = RequestInit & {
  service: string;
  timeoutMs?: number;
  retries?: number;
  retryDelayMs?: number;
  retryableStatusCodes?: number[];
};

function waitMs(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message.trim().length > 0) {
    return error.message;
  }
  return String(error);
}

function isRetryableError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }

  if (error.name === "AbortError") {
    return true;
  }

  const message = error.message.toLowerCase();
  return (
    message.includes("fetch failed") ||
    message.includes("network") ||
    message.includes("socket") ||
    message.includes("econnreset") ||
    message.includes("etimedout") ||
    message.includes("timeout")
  );
}

export async function externalFetch(url: string, options: ExternalFetchOptions): Promise<Response> {
  const timeoutMs = Math.max(1_000, options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  const retries = Math.max(0, options.retries ?? DEFAULT_RETRIES);
  const retryDelayMs = Math.max(50, options.retryDelayMs ?? DEFAULT_RETRY_DELAY_MS);
  const retryableStatusCodes = new Set([
    ...DEFAULT_RETRYABLE_STATUS_CODES,
    ...(options.retryableStatusCodes ?? [])
  ]);

  const { service, timeoutMs: _timeoutMs, retries: _retries, retryDelayMs: _retryDelayMs, retryableStatusCodes: _codes, ...init } = options;

  let lastErrorMessage = "";

  for (let attempt = 0; attempt <= retries; attempt += 1) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort("request_timeout"), timeoutMs);

    try {
      const response = await fetch(url, {
        ...init,
        signal: controller.signal
      });

      clearTimeout(timeout);

      if (retryableStatusCodes.has(response.status) && attempt < retries) {
        await waitMs(retryDelayMs * (attempt + 1));
        continue;
      }

      return response;
    } catch (error) {
      clearTimeout(timeout);
      lastErrorMessage = errorMessage(error);
      const shouldRetry = isRetryableError(error) && attempt < retries;
      if (shouldRetry) {
        await waitMs(retryDelayMs * (attempt + 1));
        continue;
      }
      throw new Error(`${service} request failed: ${lastErrorMessage}`);
    }
  }

  throw new Error(`${options.service} request failed after ${retries + 1} attempts: ${lastErrorMessage || "Unknown network error"}`);
}

