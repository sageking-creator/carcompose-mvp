import { getEnv } from "@/lib/env";

const RUNPOD_GRAPHQL_ENDPOINT = "https://api.runpod.io/graphql";

type GraphqlError = { message: string };

type GraphqlResponse<T> = {
  data?: T;
  errors?: GraphqlError[];
};

class RunpodGraphqlRequestError extends Error {
  status: number;
  messages: string[];

  constructor(status: number, messages: string[]) {
    super(`RunPod GraphQL error (${status}): ${messages.join("; ")}`);
    this.name = "RunpodGraphqlRequestError";
    this.status = status;
    this.messages = messages;
  }
}

function isSchemaCompatibilityError(error: unknown): boolean {
  if (!(error instanceof RunpodGraphqlRequestError)) {
    return false;
  }

  const text = error.messages.join(" ").toLowerCase();
  return (
    text.includes("unknown type") ||
    text.includes("cannot query field") ||
    text.includes("unknown argument") ||
    text.includes("did you mean")
  );
}

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
    const messages = json.errors?.map((item) => item.message) ?? [response.statusText || "Request failed"];
    throw new RunpodGraphqlRequestError(response.status, messages);
  }

  if (!json.data) {
    throw new RunpodGraphqlRequestError(response.status, ["Missing data in response"]);
  }

  return json.data;
}

async function runWithSchemaFallback<T>(attempts: Array<() => Promise<T>>): Promise<T> {
  let lastSchemaError: unknown = null;

  for (let index = 0; index < attempts.length; index += 1) {
    const attempt = attempts[index];
    try {
      return await attempt();
    } catch (error) {
      const shouldTryNext = isSchemaCompatibilityError(error) && index < attempts.length - 1;
      if (shouldTryNext) {
        lastSchemaError = error;
        continue;
      }
      throw error;
    }
  }

  if (lastSchemaError) {
    throw lastSchemaError;
  }
  throw new Error("RunPod GraphQL fallback exhausted without a result.");
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
  const templates = await runWithSchemaFallback<Array<{ id: string; name: string }>>([
    async () => {
      const data = await runpodGraphql<{
        myself?: { podTemplates?: Array<{ id: string; name: string }> };
      }>(
        `query ListPodTemplates {
          myself {
            podTemplates {
              id
              name
            }
          }
        }`
      );
      return data.myself?.podTemplates ?? [];
    },
    async () => {
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
      return data.myself?.templates ?? [];
    }
  ]);

  const match = templates.find((template) => template.name === name);
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

  return runWithSchemaFallback<string>([
    async () => {
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
    },
    async () => {
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
    },
    async () => {
      const data = await runpodGraphql<{
        saveNetworkVolume: { id: string };
      }>(
        `mutation SaveNetworkVolume($name: String!, $size: Int!, $dataCenterId: String!) {
          saveNetworkVolume(input: { name: $name, size: $size, dataCenterId: $dataCenterId }) {
            id
          }
        }`,
        {
          name: params.name,
          size: params.sizeGb,
          dataCenterId: params.datacenterId
        }
      );
      return data.saveNetworkVolume.id;
    }
  ]);
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

  return runWithSchemaFallback<string>([
    async () => {
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
    },
    async () => {
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
    },
    async () => {
      const data = await runpodGraphql<{
        saveTemplate: { id: string };
      }>(
        `mutation SaveTemplate(
          $name: String!,
          $imageName: String!,
          $volumeInGb: Int!,
          $volumeMountPath: String!,
          $env: [EnvInput!],
          $registryUsername: String,
          $registryPassword: String
        ) {
          saveTemplate(
            input: {
              name: $name,
              imageName: $imageName,
              containerDiskInGb: 20,
              volumeInGb: $volumeInGb,
              volumeMountPath: $volumeMountPath,
              env: $env,
              isServerless: true,
              registryAuthUsername: $registryUsername,
              registryAuthPassword: $registryPassword
            }
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
      return data.saveTemplate.id;
    }
  ]);
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

  const endpointPayload = {
    name: params.name,
    templateId: params.templateId,
    networkVolumeId: params.volumeId,
    workersMin: params.workersMin,
    workersMax: params.workersMax,
    idleTimeout: params.idleTimeout,
    scalerType: "QUEUE_DELAY",
    scalerValue: 4,
    executionTimeout: params.executionTimeoutMs
  };

  return runWithSchemaFallback<string>([
    async () => {
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
            ...endpointPayload,
            gpuIds: params.gpuType
          }
        }
      );
      return data.createEndpoint.id;
    },
    async () => {
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
            ...endpointPayload,
            gpuIds: [params.gpuType]
          }
        }
      );
      return data.createEndpoint.id;
    },
    async () => {
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
    },
    async () => {
      const data = await runpodGraphql<{
        createEndpoint: { id: string };
      }>(
        `mutation CreateEndpoint(
          $name: String!,
          $templateId: String!,
          $gpuIds: [String!]!,
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
          gpuIds: [params.gpuType],
          networkVolumeId: params.volumeId,
          workersMin: params.workersMin,
          workersMax: params.workersMax,
          idleTimeout: params.idleTimeout,
          executionTimeout: params.executionTimeoutMs
        }
      );
      return data.createEndpoint.id;
    },
    async () => {
      const data = await runpodGraphql<{
        saveEndpoint: { id: string };
      }>(
        `mutation SaveEndpoint(
          $name: String!,
          $templateId: String!,
          $gpuIds: String!,
          $networkVolumeId: String!,
          $workersMin: Int!,
          $workersMax: Int!,
          $idleTimeout: Int!,
          $executionTimeout: Int!
        ) {
          saveEndpoint(
            input: {
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
            }
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
      return data.saveEndpoint.id;
    },
    async () => {
      const data = await runpodGraphql<{
        saveEndpoint: { id: string };
      }>(
        `mutation SaveEndpoint(
          $name: String!,
          $templateId: String!,
          $gpuIds: [String!]!,
          $networkVolumeId: String!,
          $workersMin: Int!,
          $workersMax: Int!,
          $idleTimeout: Int!,
          $executionTimeout: Int!
        ) {
          saveEndpoint(
            input: {
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
            }
          ) {
            id
          }
        }`,
        {
          name: params.name,
          templateId: params.templateId,
          gpuIds: [params.gpuType],
          networkVolumeId: params.volumeId,
          workersMin: params.workersMin,
          workersMax: params.workersMax,
          idleTimeout: params.idleTimeout,
          executionTimeout: params.executionTimeoutMs
        }
      );
      return data.saveEndpoint.id;
    }
  ]);
}
