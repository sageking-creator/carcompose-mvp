import { getEnv } from "@/lib/env";
import { externalFetch } from "@/lib/external-fetch";

const RUNPOD_REST_BASE_URL = "https://rest.runpod.io/v1";

export type RunpodEndpointRest = {
  id: string;
  name?: string;
  templateId?: string;
  networkVolumeId?: string;
  networkVolumeIds?: string[];
  gpuTypeIds?: string[];
  workersMin?: number;
  workersMax?: number;
  workersStandby?: number;
  idleTimeout?: number;
  executionTimeoutMs?: number;
  scalerType?: string;
  scalerValue?: number;
  dataCenterIds?: string[];
  createdAt?: string;
};

async function runpodRest<T>(path: string, init?: RequestInit): Promise<T> {
  const env = getEnv();
  const response = await externalFetch(`${RUNPOD_REST_BASE_URL}${path}`, {
    service: "RunPod REST",
    timeoutMs: 20_000,
    retries: 2,
    ...init,
    headers: {
      Authorization: `Bearer ${env.RUNPOD_API_KEY}`,
      ...(init?.headers ?? {})
    }
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`RunPod REST error (${response.status}): ${text || response.statusText}`);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export async function listRunpodEndpointsRest(params?: {
  includeWorkers?: boolean;
  includeTemplate?: boolean;
}): Promise<RunpodEndpointRest[]> {
  const search = new URLSearchParams();
  if (typeof params?.includeWorkers === "boolean") {
    search.set("includeWorkers", String(params.includeWorkers));
  }
  if (typeof params?.includeTemplate === "boolean") {
    search.set("includeTemplate", String(params.includeTemplate));
  }
  const suffix = search.toString() ? `?${search.toString()}` : "";
  return runpodRest<RunpodEndpointRest[]>(`/endpoints${suffix}`);
}

export async function patchRunpodEndpointRest(
  endpointId: string,
  patch: Partial<RunpodEndpointRest>
): Promise<RunpodEndpointRest> {
  return runpodRest<RunpodEndpointRest>(`/endpoints/${endpointId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch)
  });
}

export async function deleteRunpodEndpointRest(endpointId: string): Promise<void> {
  await runpodRest<void>(`/endpoints/${endpointId}`, { method: "DELETE" });
}
