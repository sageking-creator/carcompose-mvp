import { getEnv } from "@/lib/env";

const RUNPOD_GRAPHQL_ENDPOINT = "https://api.runpod.io/graphql";

type GraphqlError = { message: string };

type GraphqlResponse<T> = {
  data?: T;
  errors?: GraphqlError[];
};

async function runpodGraphql<T>(query: string, variables?: Record<string, unknown>): Promise<T> {
  const env = getEnv();
  const response = await fetch(RUNPOD_GRAPHQL_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.RUNPOD_API_KEY}`
    },
    body: JSON.stringify({ query, variables })
  });

  const json = (await response.json()) as GraphqlResponse<T>;
  if (!response.ok || json.errors?.length) {
    const message = json.errors?.map((item) => item.message).join("; ") ?? response.statusText;
    throw new Error(`RunPod GraphQL error (${response.status}): ${message}`);
  }

  if (!json.data) {
    throw new Error("RunPod GraphQL error: missing data in response");
  }

  return json.data;
}

async function tryFindVolumeByName(name: string): Promise<string | null> {
  const data = await runpodGraphql<{
    myself?: { networkVolumes?: Array<{ id: string; name: string }> };
  }>(
    `query ListNetworkVolumes {
      myself {
        networkVolumes {
          id
          name
        }
      }
    }`
  );

  const match = data.myself?.networkVolumes?.find((volume) => volume.name === name);
  return match?.id ?? null;
}

async function tryFindTemplateByName(name: string): Promise<string | null> {
  const data = await runpodGraphql<{
    myself?: { templates?: Array<{ id: string; name: string }> };
  }>(
    `query ListTemplates {
      myself {
        templates {
          id
          name
        }
      }
    }`
  );

  const match = data.myself?.templates?.find((template) => template.name === name);
  return match?.id ?? null;
}

async function tryFindEndpointByName(name: string): Promise<string | null> {
  const data = await runpodGraphql<{
    myself?: { endpoints?: Array<{ id: string; name: string }> };
  }>(
    `query ListEndpoints {
      myself {
        endpoints {
          id
          name
        }
      }
    }`
  );

  const match = data.myself?.endpoints?.find((endpoint) => endpoint.name === name);
  return match?.id ?? null;
}

export async function ensureVolume(params: {
  existingId?: string;
  name: string;
  sizeGb: number;
  datacenterId: string;
}): Promise<string> {
  if (params.existingId) {
    return params.existingId;
  }

  const discovered = await tryFindVolumeByName(params.name);
  if (discovered) {
    return discovered;
  }

  const data = await runpodGraphql<{
    createNetworkVolume: { id: string };
  }>(
    `mutation CreateNetworkVolume($input: CreateNetworkVolumeInput!) {
      createNetworkVolume(input: $input) {
        id
      }
    }`,
    {
      input: {
        name: params.name,
        size: params.sizeGb,
        dataCenterId: params.datacenterId
      }
    }
  );

  return data.createNetworkVolume.id;
}

export async function ensureTemplate(params: {
  existingId?: string;
  name: string;
  dockerImage: string;
  volumeGb: number;
  volumeMountPath: string;
  env: Array<{ key: string; value: string }>;
  registryAuth?: { username: string; password: string };
}): Promise<string> {
  if (params.existingId) {
    return params.existingId;
  }

  const discovered = await tryFindTemplateByName(params.name);
  if (discovered) {
    return discovered;
  }

  const data = await runpodGraphql<{
    createTemplate: { id: string };
  }>(
    `mutation CreateTemplate($input: CreateTemplateInput!) {
      createTemplate(input: $input) {
        id
      }
    }`,
    {
      input: {
        name: params.name,
        imageName: params.dockerImage,
        containerDiskInGb: 20,
        volumeInGb: params.volumeGb,
        volumeMountPath: params.volumeMountPath,
        env: params.env,
        isServerless: true,
        registryAuthUsername: params.registryAuth?.username,
        registryAuthPassword: params.registryAuth?.password
      }
    }
  );

  return data.createTemplate.id;
}

export async function ensureEndpoint(params: {
  existingId?: string;
  name: string;
  templateId: string;
  volumeId: string;
  gpuType: string;
  workersMin: number;
  workersMax: number;
  idleTimeout: number;
  executionTimeoutMs: number;
}): Promise<string> {
  if (params.existingId) {
    return params.existingId;
  }

  const discovered = await tryFindEndpointByName(params.name);
  if (discovered) {
    return discovered;
  }

  const data = await runpodGraphql<{
    createEndpoint: { id: string };
  }>(
    `mutation CreateEndpoint($input: CreateEndpointInput!) {
      createEndpoint(input: $input) {
        id
      }
    }`,
    {
      input: {
        name: params.name,
        templateId: params.templateId,
        gpuIds: params.gpuType,
        networkVolumeId: params.volumeId,
        workersMin: params.workersMin,
        workersMax: params.workersMax,
        idleTimeout: params.idleTimeout,
        scalerType: "QUEUE_DELAY",
        scalerValue: 4,
        executionTimeout: params.executionTimeoutMs
      }
    }
  );

  return data.createEndpoint.id;
}
