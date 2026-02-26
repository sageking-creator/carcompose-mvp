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

function isUniqueNameConstraintError(error: unknown): boolean {
  if (!(error instanceof RunpodGraphqlRequestError)) {
    return false;
  }

  const text = error.messages.join(" ").toLowerCase();
  return (
    text.includes("name must be unique") ||
    text.includes("already exists") ||
    text.includes("duplicate")
  );
}

async function waitMs(ms: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms));
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
    text.includes("did you mean") ||
    text.includes("is not defined by type") ||
    text.includes("used in position expecting type") ||
    text.includes("of required type")
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

  const text = await response.text();
  let json: GraphqlResponse<T>;
  try {
    json = JSON.parse(text) as GraphqlResponse<T>;
  } catch {
    throw new RunpodGraphqlRequestError(response.status, [text || response.statusText || "Non-JSON response"]);
  }
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

async function tryFindVolumeByNameAndDatacenter(
  name: string,
  datacenterId: string
): Promise<string | null> {
  const data = await runpodGraphql<{
    myself?: { networkVolumes?: Array<{ id: string; name: string; dataCenterId?: string | null }> };
  }>(
    `query ListNetworkVolumes {
      myself {
        networkVolumes {
          id
          name
          dataCenterId
        }
      }
    }`
  );

  const match = data.myself?.networkVolumes?.find(
    (volume) => volume.name === name && volume.dataCenterId === datacenterId
  );
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

async function tryFindTemplateByNameWithRetries(
  name: string,
  retries = 5,
  delayMs = 300
): Promise<string | null> {
  for (let attempt = 0; attempt < retries; attempt += 1) {
    const templateId = await tryFindTemplateByName(name);
    if (templateId) {
      return templateId;
    }
    if (attempt < retries - 1) {
      await waitMs(delayMs * (attempt + 1));
    }
  }
  return null;
}

async function tryFindRegistryAuthByName(name: string): Promise<string | null> {
  const registryCreds = await runWithSchemaFallback<Array<{ id: string; name: string }>>([
    async () => {
      const data = await runpodGraphql<{
        myself?: { containerRegistryCreds?: Array<{ id: string; name: string }> };
      }>(
        `query ListContainerRegistryCreds {
          myself {
            containerRegistryCreds {
              id
              name
            }
          }
        }`
      );
      return data.myself?.containerRegistryCreds ?? [];
    },
    async () => {
      const data = await runpodGraphql<{
        myself?: { registryAuths?: Array<{ id: string; name: string }> };
      }>(
        `query ListRegistryAuths {
          myself {
            registryAuths {
              id
              name
            }
          }
        }`
      );
      return data.myself?.registryAuths ?? [];
    }
  ]);

  const match = registryCreds.find((cred) => cred.name === name);
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

  const discovered = await runWithSchemaFallback<string | null>([
    async () => tryFindVolumeByNameAndDatacenter(params.name, params.datacenterId),
    async () => tryFindVolumeByName(params.name)
  ]);
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

export async function ensureRegistryAuth(params: {
  existingId?: string;
  name: string;
  username: string;
  password: string;
}): Promise<string> {
  if (params.existingId) {
    return params.existingId;
  }

  const discovered = await tryFindRegistryAuthByName(params.name);
  if (discovered) {
    return discovered;
  }

  return runWithSchemaFallback<string>([
    async () => {
      const data = await runpodGraphql<{
        saveRegistryAuth: { id: string };
      }>(
        `mutation SaveRegistryAuth($input: SaveRegistryAuthInput!) {
          saveRegistryAuth(input: $input) {
            id
          }
        }`,
        {
          input: {
            name: params.name,
            username: params.username,
            password: params.password
          }
        }
      );
      return data.saveRegistryAuth.id;
    },
    async () => {
      const data = await runpodGraphql<{
        saveRegistryAuth: { id: string };
      }>(
        `mutation SaveRegistryAuth($name: String!, $username: String!, $password: String!) {
          saveRegistryAuth(input: { name: $name, username: $username, password: $password }) {
            id
          }
        }`,
        {
          name: params.name,
          username: params.username,
          password: params.password
        }
      );
      return data.saveRegistryAuth.id;
    }
  ]);
}

export async function ensureTemplate(params: {
  existingId?: string;
  name: string;
  dockerImage: string;
  volumeMountPath: string;
  env: Array<{ key: string; value: string }>;
  registryAuthId?: string;
}): Promise<string> {
  if (params.existingId) {
    return params.existingId;
  }

  const discovered = await tryFindTemplateByName(params.name);
  if (discovered) {
    return discovered;
  }

  const baseTemplateInput = {
    name: params.name,
    imageName: params.dockerImage,
    containerDiskInGb: 20,
    dockerArgs: "",
    volumeInGb: 0,
    volumeMountPath: params.volumeMountPath,
    env: params.env,
    isServerless: true
  };

  const templateInput = params.registryAuthId
    ? { ...baseTemplateInput, containerRegistryAuthId: params.registryAuthId }
    : baseTemplateInput;

  try {
    return await runWithSchemaFallback<string>([
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
            input: templateInput
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
            $env: [EnvironmentVariableInput]!,
            $containerRegistryAuthId: String
          ) {
            createTemplate(
              name: $name,
              imageName: $imageName,
              containerDiskInGb: 20,
              dockerArgs: "",
              volumeInGb: $volumeInGb,
              volumeMountPath: $volumeMountPath,
              env: $env,
              isServerless: true,
              containerRegistryAuthId: $containerRegistryAuthId
            ) {
              id
            }
          }`,
          {
            name: params.name,
            imageName: params.dockerImage,
            volumeInGb: 0,
            volumeMountPath: params.volumeMountPath,
            env: params.env,
            containerRegistryAuthId: params.registryAuthId
          }
        );
        return data.createTemplate.id;
      },
      async () => {
        const data = await runpodGraphql<{
          saveTemplate: { id: string };
        }>(
          `mutation SaveTemplate($input: SaveTemplateInput!) {
            saveTemplate(input: $input) {
              id
            }
          }`,
          {
            input: templateInput
          }
        );
        return data.saveTemplate.id;
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
            $env: [EnvironmentVariableInput]!
          ) {
            saveTemplate(
              input: {
                name: $name,
                imageName: $imageName,
                containerDiskInGb: 20,
                dockerArgs: "",
                volumeInGb: $volumeInGb,
                volumeMountPath: $volumeMountPath,
                env: $env,
                isServerless: true
              }
            ) {
              id
            }
          }`,
          {
            name: params.name,
            imageName: params.dockerImage,
            volumeInGb: 0,
            volumeMountPath: params.volumeMountPath,
            env: params.env
          }
        );
        return data.saveTemplate.id;
      }
    ]);
  } catch (error) {
    if (isUniqueNameConstraintError(error)) {
      const existingTemplateId = await tryFindTemplateByNameWithRetries(params.name);
      if (existingTemplateId) {
        return existingTemplateId;
      }
    }
    throw error;
  }
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

  const endpointPayloadBase = {
    name: params.name,
    templateId: params.templateId,
    networkVolumeId: params.volumeId,
    workersMin: params.workersMin,
    workersMax: params.workersMax,
    idleTimeout: params.idleTimeout,
    scalerType: "QUEUE_DELAY",
    scalerValue: 4
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
            ...endpointPayloadBase,
            gpuIds: params.gpuType,
            executionTimeoutMs: params.executionTimeoutMs
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
            ...endpointPayloadBase,
            gpuIds: params.gpuType,
            executionTimeout: params.executionTimeoutMs
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
            ...endpointPayloadBase,
            gpuIds: [params.gpuType],
            executionTimeoutMs: params.executionTimeoutMs
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
            ...endpointPayloadBase,
            gpuIds: [params.gpuType],
            executionTimeout: params.executionTimeoutMs
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
          $executionTimeoutMs: Int!
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
            executionTimeoutMs: $executionTimeoutMs
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
          executionTimeoutMs: params.executionTimeoutMs
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
          $executionTimeoutMs: Int!
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
            executionTimeoutMs: $executionTimeoutMs
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
          executionTimeoutMs: params.executionTimeoutMs
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
          $executionTimeoutMs: Int!
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
              executionTimeoutMs: $executionTimeoutMs
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
          executionTimeoutMs: params.executionTimeoutMs
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
          $executionTimeoutMs: Int!
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
              executionTimeoutMs: $executionTimeoutMs
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
          executionTimeoutMs: params.executionTimeoutMs
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

export type RunpodGpuMarketItem = {
  id: string;
  displayName: string;
  memoryInGb: number;
  lowestPrice: null | {
    uninterruptablePrice: number | null;
    minimumBidPrice: number | null;
    stockStatus: string | null;
    maxUnreservedGpuCount: number | null;
  };
};

export async function listDataCenterIds(): Promise<string[]> {
  const data = await runpodGraphql<{ dataCenters: Array<{ id: string }> }>(
    `query ListDataCenters {
      dataCenters {
        id
      }
    }`
  );
  return data.dataCenters.map((item) => item.id);
}

export async function listGpuMarketForDatacenter(params: {
  datacenterId: string;
  secureCloud: boolean;
}): Promise<RunpodGpuMarketItem[]> {
  const data = await runpodGraphql<{
    gpuTypes: Array<RunpodGpuMarketItem>;
  }>(
    `query GpuMarket($datacenterId: String, $secureCloud: Boolean!) {
      gpuTypes {
        id
        displayName
        memoryInGb
        lowestPrice(
          input: {
            secureCloud: $secureCloud
            gpuCount: 1
            globalNetwork: false
            minDisk: 0
            minMemoryInGb: 8
            minVcpuCount: 2
            dataCenterId: $datacenterId
            compliance: null
          }
        ) {
          uninterruptablePrice
          minimumBidPrice
          stockStatus
          maxUnreservedGpuCount
        }
      }
    }`,
    { datacenterId: params.datacenterId, secureCloud: params.secureCloud }
  );

  return data.gpuTypes;
}

export type RunpodGpuLowestPrice = null | {
  uninterruptablePrice: number | null;
  minimumBidPrice: number | null;
  stockStatus: string | null;
  maxUnreservedGpuCount: number | null;
};

export async function getGpuLowestPrice(params: {
  gpuTypeId: string;
  datacenterId: string;
  secureCloud: boolean;
}): Promise<{ memoryInGb: number | null; lowestPrice: RunpodGpuLowestPrice }> {
  const data = await runpodGraphql<{
    gpuTypes: Array<{ memoryInGb: number | null; lowestPrice: RunpodGpuLowestPrice }>;
  }>(
    `query GpuLowestPrice($id: String!, $datacenterId: String, $secureCloud: Boolean!) {
      gpuTypes(input: { id: $id }) {
        memoryInGb
        lowestPrice(
          input: {
            secureCloud: $secureCloud
            gpuCount: 1
            globalNetwork: false
            minDisk: 0
            minMemoryInGb: 8
            minVcpuCount: 2
            dataCenterId: $datacenterId
            compliance: null
          }
        ) {
          uninterruptablePrice
          minimumBidPrice
          stockStatus
          maxUnreservedGpuCount
        }
      }
    }`,
    { id: params.gpuTypeId, datacenterId: params.datacenterId, secureCloud: params.secureCloud }
  );

  const first = data.gpuTypes[0];
  return {
    memoryInGb: first?.memoryInGb ?? null,
    lowestPrice: first?.lowestPrice ?? null
  };
}

export async function deleteNetworkVolume(volumeId: string): Promise<void> {
  await runWithSchemaFallback<void>([
    async () => {
      await runpodGraphql<{ deleteNetworkVolume: null }>(
        `mutation DeleteNetworkVolume($id: String!) {
          deleteNetworkVolume(input: { id: $id })
        }`,
        { id: volumeId }
      );
    },
    async () => {
      await runpodGraphql<{ deleteNetworkVolume: null }>(
        `mutation DeleteNetworkVolume($id: String!) {
          deleteNetworkVolume(id: $id)
        }`,
        { id: volumeId }
      );
    }
  ]);
}
