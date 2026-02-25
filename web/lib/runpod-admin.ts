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
    `mutation CreateNetworkVolume($name: String!, $size: Int!, $dataCenterId: String!) {
      createNetworkVolume(name: $name, size: $size, dataCenterId: $dataCenterId) {
        id
      }
    }`,
    {
      name: params.name,
      size: params.sizeGb,
      dataCenterId: params.datacenterId
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
    `mutation CreateTemplate(
      $name: String!,
      $imageName: String!,
      $volumeInGb: Int!,
      $volumeMountPath: String!,
      $env: [EnvInput!],
      $registryUsername: String,
      $registryPassword: String
    ) {
      createTemplate(
        name: $name,
        imageName: $imageName,
        containerDiskInGb: 20,
        volumeInGb: $volumeInGb,
        volumeMountPath: $volumeMountPath,
        env: $env,
        isServerless: true,
        registryAuthUsername: $registryUsername,
        registryAuthPassword: $registryPassword
      ) {
        id
      }
    }`,
    {
      name: params.name,
      imageName: params.dockerImage,
      volumeInGb: params.volumeGb,
      volumeMountPath: params.volumeMountPath,
      env: params.env,
      registryUsername: params.registryAuth?.username,
      registryPassword: params.registryAuth?.password
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
    `mutation CreateEndpoint(
      $name: String!,
      $templateId: String!,
      $gpuIds: String!,
      $networkVolumeId: String!,
      $workersMin: Int!,
      $workersMax: Int!,
      $idleTimeout: Int!,
      $executionTimeout: Int!
    ) {
      createEndpoint(
        name: $name,
        templateId: $templateId,
        gpuIds: $gpuIds,
        networkVolumeId: $networkVolumeId,
        workersMin: $workersMin,
        workersMax: $workersMax,
        idleTimeout: $idleTimeout,
        scalerType: "QUEUE_DELAY",
        scalerValue: 4,
        executionTimeout: $executionTimeout
      ) {
        id
      }
    }`,
    {
      name: params.name,
      templateId: params.templateId,
      gpuIds: params.gpuType,
      networkVolumeId: params.volumeId,
      workersMin: params.workersMin,
      workersMax: params.workersMax,
      idleTimeout: params.idleTimeout,
      executionTimeout: params.executionTimeoutMs
    }
  );

  return data.createEndpoint.id;
}
