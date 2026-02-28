import { externalFetch } from "@/lib/external-fetch";

const FAL_QUEUE_URL = "https://queue.fal.run/fal-ai/birefnet/v2";

type FalStatus = "IN_QUEUE" | "IN_PROGRESS" | "COMPLETED" | "FAILED" | "CANCELLED";

type FalSubmitResponse = {
  request_id?: string;
  status_url?: string;
  response_url?: string;
  status?: string;
};

function waitMs(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function toRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "object" || value === null) {
    return null;
  }
  return value as Record<string, unknown>;
}

function extractStatus(payload: unknown): FalStatus {
  const record = toRecord(payload);
  const raw = String(record?.status ?? "").toUpperCase();
  if (raw === "COMPLETED" || raw === "FAILED" || raw === "CANCELLED" || raw === "IN_QUEUE" || raw === "IN_PROGRESS") {
    return raw;
  }
  return "IN_PROGRESS";
}

function extractFailureReason(payload: unknown): string | null {
  const record = toRecord(payload);
  if (!record) {
    return null;
  }
  const directError = record.error;
  if (typeof directError === "string" && directError.trim().length > 0) {
    return directError;
  }

  const errorRecord = toRecord(record.error);
  if (errorRecord) {
    const message = errorRecord.message;
    if (typeof message === "string" && message.trim().length > 0) {
      return message;
    }
  }

  return null;
}

function extractMaskUrl(payload: unknown): string | null {
  const record = toRecord(payload);
  if (!record) {
    return null;
  }

  const candidates: unknown[] = [
    record.mask_image,
    toRecord(record.response)?.mask_image,
    toRecord(record.output)?.mask_image,
    toRecord(toRecord(record.data)?.output)?.mask_image,
    record.mask,
    toRecord(record.response)?.mask,
    toRecord(record.output)?.mask
  ];

  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.startsWith("http")) {
      return candidate;
    }
    const candidateRecord = toRecord(candidate);
    const url = candidateRecord?.url;
    if (typeof url === "string" && url.startsWith("http")) {
      return url;
    }
  }

  return null;
}

async function parseJsonSafe(response: Response): Promise<unknown> {
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.toLowerCase().includes("application/json")) {
    const text = await response.text();
    return { raw: text };
  }
  return response.json();
}

async function falRequest(
  path: string,
  apiKey: string,
  init: RequestInit & { timeoutMs?: number; retries?: number }
): Promise<Response> {
  return externalFetch(path, {
    service: "fal.ai BiRefNet v2",
    timeoutMs: init.timeoutMs ?? 30_000,
    retries: init.retries ?? 2,
    retryDelayMs: 500,
    ...init,
    headers: {
      Authorization: `Key ${apiKey}`,
      "Content-Type": "application/json",
      ...(init.headers ?? {})
    }
  });
}

export type FalBirefnetConfig = {
  apiKey: string;
  imageUrl: string;
  model: string;
  operatingResolution: string;
  refineForeground: boolean;
  timeoutSeconds: number;
};

export type FalBirefnetResult = {
  requestId: string;
  maskUrl: string;
  maskBytes: Buffer;
  contentType: string;
};

export async function generateFalBirefnetMask(config: FalBirefnetConfig): Promise<FalBirefnetResult> {
  const submitResponse = await falRequest(FAL_QUEUE_URL, config.apiKey, {
    method: "POST",
    body: JSON.stringify({
      model: config.model,
      operating_resolution: config.operatingResolution,
      refine_foreground: config.refineForeground,
      output_mask: true,
      output_format: "png",
      image_url: config.imageUrl
    })
  });

  if (!submitResponse.ok) {
    const body = await submitResponse.text();
    throw new Error(`fal submit failed (${submitResponse.status}): ${body || submitResponse.statusText}`);
  }

  const submitPayload = (await parseJsonSafe(submitResponse)) as FalSubmitResponse;
  const requestId = submitPayload.request_id;
  if (!requestId) {
    throw new Error("fal submit response missing request_id");
  }

  const statusUrl = submitPayload.status_url ?? `${FAL_QUEUE_URL}/requests/${requestId}/status`;
  const responseUrl = submitPayload.response_url ?? `${FAL_QUEUE_URL}/requests/${requestId}`;
  const deadline = Date.now() + Math.max(30, config.timeoutSeconds) * 1000;

  for (;;) {
    if (Date.now() > deadline) {
      throw new Error(`fal request timed out after ${config.timeoutSeconds}s`);
    }

    const statusResponse = await falRequest(statusUrl, config.apiKey, {
      method: "GET",
      timeoutMs: 25_000,
      retries: 2
    });

    if (!statusResponse.ok) {
      const body = await statusResponse.text();
      throw new Error(`fal status failed (${statusResponse.status}): ${body || statusResponse.statusText}`);
    }

    const statusPayload = await parseJsonSafe(statusResponse);
    const status = extractStatus(statusPayload);
    if (status === "COMPLETED") {
      break;
    }
    if (status === "FAILED" || status === "CANCELLED") {
      const reason = extractFailureReason(statusPayload) ?? "Unknown fal failure";
      throw new Error(`fal request failed: ${reason}`);
    }

    await waitMs(1800);
  }

  const resultResponse = await falRequest(responseUrl, config.apiKey, {
    method: "GET",
    timeoutMs: 25_000,
    retries: 2
  });
  if (!resultResponse.ok) {
    const body = await resultResponse.text();
    throw new Error(`fal result fetch failed (${resultResponse.status}): ${body || resultResponse.statusText}`);
  }

  const resultPayload = await parseJsonSafe(resultResponse);
  const maskUrl = extractMaskUrl(resultPayload);
  if (!maskUrl) {
    throw new Error("fal result missing mask URL");
  }

  const maskDownload = await externalFetch(maskUrl, {
    service: "fal.ai mask download",
    method: "GET",
    timeoutMs: 60_000,
    retries: 2
  });

  if (!maskDownload.ok) {
    const body = await maskDownload.text();
    throw new Error(`fal mask download failed (${maskDownload.status}): ${body || maskDownload.statusText}`);
  }

  const contentType = (maskDownload.headers.get("content-type") ?? "image/png").split(";")[0].trim();
  const bytes = Buffer.from(await maskDownload.arrayBuffer());
  if (bytes.byteLength === 0) {
    throw new Error("fal mask download returned empty body");
  }

  return {
    requestId,
    maskUrl,
    maskBytes: bytes,
    contentType: contentType || "image/png"
  };
}
