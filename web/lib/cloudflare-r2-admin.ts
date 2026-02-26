import { getEnv } from "@/lib/env";
import { externalFetch } from "@/lib/external-fetch";

const CLOUDFLARE_API = "https://api.cloudflare.com/client/v4";

type CloudflareResponse<T> = {
  success: boolean;
  errors?: Array<{ message: string }>;
  result?: T;
};

async function cloudflareFetch<T>(path: string, init: RequestInit): Promise<CloudflareResponse<T>> {
  const env = getEnv();
  const response = await externalFetch(`${CLOUDFLARE_API}${path}`, {
    service: "Cloudflare R2 API",
    timeoutMs: 20_000,
    retries: 2,
    ...init,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.CLOUDFLARE_API_TOKEN}`,
      ...(init.headers ?? {})
    }
  });

  if (response.status === 404) {
    return { success: false, errors: [{ message: "Not Found" }] };
  }

  const json = (await response.json()) as CloudflareResponse<T>;
  if (!json.success && response.status >= 400) {
    const msg = json.errors?.map((item) => item.message).join("; ") ?? response.statusText;
    throw new Error(`Cloudflare API error (${response.status}): ${msg}`);
  }

  return json;
}

export async function ensureBucketExists(bucketName: string): Promise<void> {
  const env = getEnv();
  const checkPath = `/accounts/${env.CLOUDFLARE_ACCOUNT_ID}/r2/buckets/${bucketName}`;
  const check = await cloudflareFetch<unknown>(checkPath, { method: "GET" });
  if (check.success) {
    return;
  }

  try {
    await cloudflareFetch<unknown>(`/accounts/${env.CLOUDFLARE_ACCOUNT_ID}/r2/buckets`, {
      method: "POST",
      body: JSON.stringify({ name: bucketName })
    });
  } catch (error) {
    const message = String(error).toLowerCase();
    if (message.includes("already") && message.includes("exist")) {
      const retry = await cloudflareFetch<unknown>(checkPath, { method: "GET" });
      if (retry.success) {
        return;
      }
      throw new Error(
        `R2 bucket name "${bucketName}" already exists (or cannot be created with this token). ` +
          `Set a different R2_BUCKET_NAME or verify your Cloudflare token has R2 write permissions.`
      );
    }
    throw error;
  }
}
