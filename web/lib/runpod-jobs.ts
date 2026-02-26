import { getEnv } from "@/lib/env";
import { externalFetch } from "@/lib/external-fetch";

export type RunpodJobStatus = {
  status: string;
  output?: unknown;
  error?: string;
};

export type RunpodHealth = {
  jobs?: {
    completed?: number;
    failed?: number;
    inProgress?: number;
    inQueue?: number;
    retried?: number;
  };
  workers?: {
    idle?: number;
    initializing?: number;
    ready?: number;
    running?: number;
    throttled?: number;
    unhealthy?: number;
  };
};

export type RunpodRequestItem = {
  id: string;
  status?: string;
  delayTime?: number;
};

export type RunpodRequests = {
  requests?: RunpodRequestItem[];
};

function getRunpodBaseUrl(endpointId: string): string {
  return `https://api.runpod.ai/v2/${endpointId}`;
}

export async function submitRunpodJob(endpointId: string, input: Record<string, unknown>): Promise<string> {
  const env = getEnv();
  const response = await externalFetch(`${getRunpodBaseUrl(endpointId)}/run`, {
    service: "RunPod run",
    timeoutMs: 20_000,
    retries: 2,
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
  const response = await externalFetch(`${getRunpodBaseUrl(endpointId)}/status/${jobId}`, {
    service: "RunPod status",
    timeoutMs: 20_000,
    retries: 2,
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

export async function getRunpodHealth(endpointId: string): Promise<RunpodHealth> {
  const env = getEnv();
  const response = await externalFetch(`${getRunpodBaseUrl(endpointId)}/health`, {
    service: "RunPod health",
    timeoutMs: 20_000,
    retries: 2,
    headers: {
      Authorization: `Bearer ${env.RUNPOD_API_KEY}`
    }
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`RunPod health check failed (${response.status}): ${text}`);
  }

  return (await response.json()) as RunpodHealth;
}

export async function getRunpodRequests(endpointId: string): Promise<RunpodRequests> {
  const env = getEnv();
  const response = await externalFetch(`${getRunpodBaseUrl(endpointId)}/requests`, {
    service: "RunPod requests",
    timeoutMs: 20_000,
    retries: 2,
    headers: {
      Authorization: `Bearer ${env.RUNPOD_API_KEY}`
    }
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`RunPod requests check failed (${response.status}): ${text}`);
  }

  return (await response.json()) as RunpodRequests;
}

export type RunpodPurgeQueueResult = {
  removed?: number;
  status?: string;
  error?: string;
};

export async function purgeRunpodQueue(endpointId: string): Promise<RunpodPurgeQueueResult> {
  const env = getEnv();
  const response = await externalFetch(`${getRunpodBaseUrl(endpointId)}/purge-queue`, {
    service: "RunPod purge-queue",
    timeoutMs: 20_000,
    retries: 2,
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.RUNPOD_API_KEY}`
    }
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`RunPod purge queue failed (${response.status}): ${text}`);
  }

  return (await response.json()) as RunpodPurgeQueueResult;
}

export type RunpodCancelJobResult = {
  id?: string;
  status?: string;
  error?: string;
};

export async function cancelRunpodJob(endpointId: string, jobId: string): Promise<RunpodCancelJobResult> {
  const env = getEnv();
  const response = await externalFetch(`${getRunpodBaseUrl(endpointId)}/cancel/${jobId}`, {
    service: "RunPod cancel-job",
    timeoutMs: 20_000,
    retries: 2,
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.RUNPOD_API_KEY}`
    }
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`RunPod cancel job failed (${response.status}): ${text}`);
  }

  return (await response.json()) as RunpodCancelJobResult;
}
