import { getEnv } from "@/lib/env";

export type RunpodJobStatus = {
  status: string;
  output?: unknown;
  error?: string;
};

function getRunpodBaseUrl(endpointId: string): string {
  return `https://api.runpod.ai/v2/${endpointId}`;
}

export async function submitRunpodJob(endpointId: string, input: Record<string, unknown>): Promise<string> {
  const env = getEnv();
  const response = await fetch(`${getRunpodBaseUrl(endpointId)}/run`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.RUNPOD_API_KEY}`
    },
    body: JSON.stringify({ input })
  });

  const payload = (await response.json()) as { id?: string; error?: string };
  if (!response.ok || !payload.id) {
    throw new Error(payload.error ?? `RunPod job submission failed (${response.status})`);
  }

  return payload.id;
}

export async function getRunpodJobStatus(endpointId: string, jobId: string): Promise<RunpodJobStatus> {
  const env = getEnv();
  const response = await fetch(`${getRunpodBaseUrl(endpointId)}/status/${jobId}`, {
    headers: {
      Authorization: `Bearer ${env.RUNPOD_API_KEY}`
    }
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`RunPod status check failed (${response.status}): ${text}`);
  }

  return (await response.json()) as RunpodJobStatus;
}
