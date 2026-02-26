import { ensureBucketExists } from "@/lib/cloudflare-r2-admin";
import { type AppEnv } from "@/lib/env";
import { resolveGhcrImageToDigest } from "@/lib/ghcr";
import { ensureCorsRules, ensureLifecycleRules } from "@/lib/r2";
import {
  ensureEndpoint,
  ensureRegistryAuth,
  ensureTemplate,
  ensureVolume,
  deleteNetworkVolume,
  listDataCenterIds,
  listGpuMarketForDatacenter
} from "@/lib/runpod-admin";
import {
  getRunpodHealth,
  getRunpodJobStatus,
  getRunpodRequests,
  purgeRunpodQueue,
  submitRunpodJob
} from "@/lib/runpod-jobs";
import {
  deleteRunpodEndpointRest,
  listRunpodEndpointsRest,
  patchRunpodEndpointRest
} from "@/lib/runpod-endpoints-rest";
import {
  getSetupState,
  putSetupState,
  type InitJobStatus,
  type SetupState
} from "@/lib/r2-state";
import { resolveWorkerImage } from "@/lib/worker-image";
import { createHash } from "node:crypto";

export type ReadyResult = {
  ready: boolean;
  phase: "provisioning" | "downloading_models" | "ready" | "error";
  message: string;
  details: {
    bucket: string;
    workerImage?: string;
    workerImageDigest?: string;
    volumeDatacenterId?: string;
    endpointId?: string;
    initJobId?: string;
  };
};

const VOLUME_DATACENTER_FALLBACKS = ["US-NC-1", "US-TX-3"];
const SERVERLESS_DATACENTER_IDS = new Set<string>([
  "EU-RO-1",
  "CA-MTL-1",
  "EU-SE-1",
  "US-IL-1",
  "EUR-IS-1",
  "EU-CZ-1",
  "US-TX-3",
  "EUR-IS-2",
  "US-KS-2",
  "US-GA-2",
  "US-WA-1",
  "US-TX-1",
  "CA-MTL-3",
  "EU-NL-1",
  "US-TX-4",
  "US-CA-2",
  "US-NC-1",
  "OC-AU-1",
  "US-DE-1",
  "EUR-IS-3",
  "CA-MTL-2",
  "AP-JP-1",
  "EUR-NO-1",
  "EU-FR-1",
  "US-KS-3",
  "US-GA-1"
]);

const AUTOPICK_SECURE_CLOUD = true;
const AUTOPICK_MAX_DATACENTERS = 25;
const AUTOPICK_QUEUE_FAILOVER_AFTER_MS = 2 * 60 * 1000;
const AUTOPICK_QUEUE_FAILOVER_COOLDOWN_MS = 4 * 60 * 1000;
const STALE_ENDPOINT_DRAIN_RETRIES = 3;
const STALE_ENDPOINT_DRAIN_WAIT_MS = 2_000;
const VOLUME_DATACENTER_CANDIDATES = [
  // Prefer serverless-supported US zones first.
  "US-TX-3",
  "US-NC-1",
  "US-GA-2",
  "US-CA-2",
  "US-TX-1",
  "US-WA-1",
  "US-IL-1",
  "US-KS-2",
  "US-GA-1",
  "US-KS-3",
  "US-TX-4",
  "US-DE-1"
];

function getTemplateEnv(env: AppEnv): Array<{ key: string; value: string }> {
  const cacheDir = env.MODEL_CACHE_DIR;
  const hfHome = env.HF_HOME;

  return [
    { key: "MODEL_CACHE_DIR", value: cacheDir },
    { key: "HF_HOME", value: hfHome },
    { key: "TRANSFORMERS_CACHE", value: hfHome },
    { key: "BIREFNET_REPO_ID", value: env.BIREFNET_REPO_ID },
    { key: "BIREFNET_INFER_RES", value: String(env.BIREFNET_INFER_RES) },
    { key: "LIBCOM_MODEL_DIR", value: `${cacheDir}/libcom` },
    { key: "CONTROLCOM_CKPT", value: `${cacheDir}/controlcom/ControlCom_blend_harm.pth` },
    { key: "CLIP_MODEL_DIR", value: `${cacheDir}/controlcom/openai-clip-vit-large-patch14` },
    { key: "CUDA_VISIBLE_DEVICES", value: "0" },
    { key: "PYTORCH_CUDA_ALLOC_CONF", value: "max_split_size_mb:512" },
    { key: "PIPELINE_VARIANT", value: env.PIPELINE_VARIANT },
    { key: "WORKER_BUILD_ID", value: env.VERCEL_GIT_COMMIT_SHA ?? "unknown" },
    { key: "MAX_OUTPUT_LONG_EDGE", value: String(env.MAX_OUTPUT_LONG_EDGE) },
    { key: "OUTPUT_RESIZE_MODE", value: env.OUTPUT_RESIZE_MODE },
    { key: "CORE_CONTACT_SHADOW_STRENGTH", value: String(env.CORE_CONTACT_SHADOW_STRENGTH) },
    { key: "CONTACT_SHADOW_MODE", value: env.CONTACT_SHADOW_MODE },
    { key: "GLASS_NORMALIZATION_MODE", value: env.GLASS_NORMALIZATION_MODE },
    { key: "STUDIO_MODE", value: env.STUDIO_MODE },
    { key: "MAX_EDGE_HALO_MEAN_DELTA", value: String(env.MAX_EDGE_HALO_MEAN_DELTA) },
    { key: "MAX_EDGE_BAND_WIDTH_PX", value: String(env.MAX_EDGE_BAND_WIDTH_PX) },
    { key: "DEBUG_ARTIFACTS", value: env.DEBUG_ARTIFACTS ? "1" : "0" }
  ];
}

function computeProvisioningHash(input: Record<string, unknown>): string {
  return createHash("sha256").update(JSON.stringify(input)).digest("hex").slice(0, 12);
}

function mapInitStatus(status: string): InitJobStatus {
  if (status === "COMPLETED") {
    return "COMPLETED";
  }

  if (status === "FAILED" || status === "CANCELLED" || status === "TIMED_OUT") {
    return "FAILED";
  }

  return "RUNNING";
}

function readWorkerErrorMessage(output: unknown): string | null {
  if (!output || typeof output !== "object") {
    return null;
  }

  const status = (output as Record<string, unknown>).status;
  if (status !== "error") {
    return null;
  }

  const message = (output as Record<string, unknown>).message;
  return typeof message === "string" && message.trim().length ? message : "Model initialization failed.";
}

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message.trim().length > 0) {
    return error.message;
  }
  return String(error);
}

function extractWorkerImageDigest(workerImage: string | null | undefined): string | undefined {
  if (!workerImage) {
    return undefined;
  }

  const atIndex = workerImage.lastIndexOf("@");
  if (atIndex === -1) {
    return undefined;
  }

  const digest = workerImage.slice(atIndex + 1).trim();
  return digest.length > 0 ? digest : undefined;
}

function isDatacenterNotFoundError(error: unknown): boolean {
  return errorMessage(error).toLowerCase().includes("failed to find data center");
}

function normalizeRegistryAuthName(username: string): string {
  const cleaned = username.toLowerCase().replace(/[^a-z0-9-]+/g, "-").replace(/-+/g, "-");
  const trimmed = cleaned.replace(/^-+/, "").replace(/-+$/, "");
  return `carcompose-ghcr-${trimmed || "user"}`.slice(0, 63);
}

function normalizeDatacenterToken(datacenterId: string): string {
  return datacenterId.toLowerCase().replace(/[^a-z0-9-]+/g, "-");
}

type PlacementCandidate = {
  datacenterId: string;
  gpuType: string;
  memoryInGb: number;
  pricePerHour: number;
  stockStatus: string;
  maxUnreservedGpuCount: number;
};

function placementKey(datacenterId: string, gpuType: string): string {
  return `${datacenterId}|${gpuType}`;
}

function minVramGbForVariant(variant: AppEnv["PIPELINE_VARIANT"]): number {
  return variant === "full" ? 48 : 24;
}

function normalizeStockStatus(status: string | null | undefined): string {
  if (!status) {
    return "Unknown";
  }
  if (status === "High" || status === "Medium" || status === "Low") {
    return status;
  }
  return status;
}

function stockRank(status: string): number {
  if (status === "High") {
    return 0;
  }
  if (status === "Medium") {
    return 1;
  }
  if (status === "Low") {
    return 2;
  }
  return 3;
}

function pickMarketPrice(lowestPrice: {
  uninterruptablePrice: number | null;
  minimumBidPrice: number | null;
}): number | null {
  const price = lowestPrice.uninterruptablePrice ?? lowestPrice.minimumBidPrice ?? null;
  if (typeof price !== "number" || !Number.isFinite(price) || price <= 0) {
    return null;
  }
  return price;
}

function isCudaGpuTypeId(id: string): boolean {
  return id.startsWith("NVIDIA ");
}

function hasImmediateCapacity(candidate: PlacementCandidate): boolean {
  return candidate.maxUnreservedGpuCount > 0;
}

function comparePlacementCandidates(
  a: PlacementCandidate,
  b: PlacementCandidate,
  preferredGpuType?: string
): number {
  if (preferredGpuType) {
    const preferredDelta =
      Number(a.gpuType !== preferredGpuType) - Number(b.gpuType !== preferredGpuType);
    if (preferredDelta !== 0) {
      return preferredDelta;
    }
  }

  const capacityDelta = Number(!hasImmediateCapacity(a)) - Number(!hasImmediateCapacity(b));
  if (capacityDelta !== 0) {
    return capacityDelta;
  }

  const priceDelta = a.pricePerHour - b.pricePerHour;
  if (priceDelta !== 0) {
    return priceDelta;
  }

  const stockDelta = stockRank(a.stockStatus) - stockRank(b.stockStatus);
  if (stockDelta !== 0) {
    return stockDelta;
  }

  const maxDelta = b.maxUnreservedGpuCount - a.maxUnreservedGpuCount;
  if (maxDelta !== 0) {
    return maxDelta;
  }

  return b.memoryInGb - a.memoryInGb;
}

async function bestPlacementForDatacenter(
  datacenterId: string,
  minVramGb: number,
  preferredGpuType?: string
): Promise<PlacementCandidate | null> {
  const market = await listGpuMarketForDatacenter({
    datacenterId,
    secureCloud: AUTOPICK_SECURE_CLOUD
  });

  const candidates: PlacementCandidate[] = [];
  for (const gpu of market) {
    if (!isCudaGpuTypeId(gpu.id)) {
      continue;
    }

    if (typeof gpu.memoryInGb !== "number" || gpu.memoryInGb < minVramGb) {
      continue;
    }

    if (!gpu.lowestPrice) {
      continue;
    }

    const pricePerHour = pickMarketPrice({
      uninterruptablePrice: gpu.lowestPrice.uninterruptablePrice,
      minimumBidPrice: gpu.lowestPrice.minimumBidPrice
    });
    if (!pricePerHour) {
      continue;
    }

    candidates.push({
      datacenterId,
      gpuType: gpu.id,
      memoryInGb: gpu.memoryInGb,
      pricePerHour,
      stockStatus: normalizeStockStatus(gpu.lowestPrice.stockStatus),
      maxUnreservedGpuCount: gpu.lowestPrice.maxUnreservedGpuCount ?? 0
    });
  }

  if (candidates.length === 0) {
    return null;
  }

  if (preferredGpuType) {
    const preferredGpuWithCapacity = candidates
      .filter((item) => item.gpuType === preferredGpuType && hasImmediateCapacity(item))
      .sort((a, b) => comparePlacementCandidates(a, b, preferredGpuType));
    if (preferredGpuWithCapacity[0]) {
      return preferredGpuWithCapacity[0];
    }
  }

  const anyWithCapacity = candidates
    .filter((item) => hasImmediateCapacity(item))
    .sort((a, b) => comparePlacementCandidates(a, b, preferredGpuType));
  if (anyWithCapacity[0]) {
    return anyWithCapacity[0];
  }

  candidates.sort((a, b) => comparePlacementCandidates(a, b, preferredGpuType));
  return candidates[0];
}

function buildDatacenterPreferenceList(preferredDatacenterId: string, extra: string[] = []): string[] {
  const seen = new Set<string>();
  const ordered = [...extra, preferredDatacenterId, ...VOLUME_DATACENTER_CANDIDATES, ...VOLUME_DATACENTER_FALLBACKS]
    .filter(Boolean)
    .filter((dc) => SERVERLESS_DATACENTER_IDS.has(dc))
    .filter((dc) => {
      if (seen.has(dc)) {
        return false;
      }
      seen.add(dc);
      return true;
    });
  return ordered;
}

async function waitMs(ms: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function cleanupStaleCarcomposeEndpoints(params: {
  keepEndpointId?: string;
  keepEndpointName?: string;
}): Promise<void> {
  let endpoints: Array<{ id: string; name?: string }> = [];
  try {
    endpoints = await listRunpodEndpointsRest({ includeWorkers: false, includeTemplate: false });
  } catch {
    return;
  }

  const stale = endpoints.filter((endpoint) => {
    const name = endpoint.name ?? "";
    if (!name.startsWith("carcompose-pipeline-")) {
      return false;
    }
    if (params.keepEndpointId && endpoint.id === params.keepEndpointId) {
      return false;
    }
    if (params.keepEndpointName && name === params.keepEndpointName) {
      return false;
    }
    return true;
  });

  for (const endpoint of stale) {
    let hasInProgressJobs = false;
    try {
      const health = await getRunpodHealth(endpoint.id);
      const jobs = health.jobs ?? {};
      hasInProgressJobs = (jobs.inProgress ?? 0) > 0;
    } catch {
      // If health fails, continue with best-effort cleanup.
    }

    try {
      await purgeRunpodQueue(endpoint.id);
    } catch {
      // Ignore purge failures.
    }

    try {
      await patchRunpodEndpointRest(endpoint.id, { workersMin: 0, workersMax: 0 });
    } catch {
      // Ignore; deletion may still succeed.
    }

    if (hasInProgressJobs) {
      continue;
    }

    for (let attempt = 0; attempt < STALE_ENDPOINT_DRAIN_RETRIES; attempt += 1) {
      try {
        const health = await getRunpodHealth(endpoint.id);
        const inProgressCount = health.jobs?.inProgress ?? 0;
        if (inProgressCount <= 0) {
          hasInProgressJobs = false;
          break;
        }
        hasInProgressJobs = true;
      } catch {
        hasInProgressJobs = false;
        break;
      }

      await waitMs(STALE_ENDPOINT_DRAIN_WAIT_MS);
    }

    if (hasInProgressJobs) {
      continue;
    }

    try {
      await deleteRunpodEndpointRest(endpoint.id);
    } catch {
      // Ignore; endpoint may already be gone or protected.
    }
  }
}

export async function ensureReady(env: AppEnv): Promise<ReadyResult> {
  const workerImage = resolveWorkerImage(env);
  if (!workerImage.image) {
    return {
      ready: false,
      phase: "error",
      message: workerImage.reason ?? "Unable to resolve worker image.",
      details: {
        bucket: env.R2_BUCKET_NAME,
        workerImage: workerImage.image ?? undefined,
        workerImageDigest: extractWorkerImageDigest(workerImage.image ?? undefined)
      }
    };
  }

  const registryAuth =
    env.GHCR_USERNAME && env.GHCR_TOKEN
      ? { username: env.GHCR_USERNAME, password: env.GHCR_TOKEN }
      : undefined;

  const resolvedWorkerImage = await resolveGhcrImageToDigest(workerImage.image, registryAuth);
  const resolvedWorkerImageDigest = extractWorkerImageDigest(resolvedWorkerImage);
  const registryAuthName = registryAuth ? normalizeRegistryAuthName(registryAuth.username) : undefined;
  let registryAuthId: string | undefined = undefined;
  if (registryAuth) {
    try {
      registryAuthId = await ensureRegistryAuth({
        name: registryAuthName ?? "carcompose-ghcr-user",
        username: registryAuth.username,
        password: registryAuth.password
      });
    } catch (error) {
      throw new Error(`RunPod registry auth provisioning failed: ${errorMessage(error)}`);
    }
  }

  await ensureBucketExists(env.R2_BUCKET_NAME);
  await ensureCorsRules(env.R2_BUCKET_NAME);
  await ensureLifecycleRules(env.R2_BUCKET_NAME);

  const templateEnv = getTemplateEnv(env)
    .slice()
    .sort((a, b) => a.key.localeCompare(b.key));
  const minVramGb = minVramGbForVariant(env.PIPELINE_VARIANT);
  const preferredGpuType = env.RUNPOD_GPU_TYPE;

  let setup: SetupState =
    (await getSetupState()) ??
    (await putSetupState({
      bucketName: env.R2_BUCKET_NAME,
      workerImage: resolvedWorkerImage,
      workerImageDigest: resolvedWorkerImageDigest,
      initJobStatus: "NOT_STARTED"
    }));

  if (
    setup.bucketName !== env.R2_BUCKET_NAME ||
    setup.workerImage !== resolvedWorkerImage ||
    setup.workerImageDigest !== resolvedWorkerImageDigest
  ) {
    setup = await putSetupState({
      bucketName: env.R2_BUCKET_NAME,
      workerImage: resolvedWorkerImage,
      workerImageDigest: resolvedWorkerImageDigest
    });
  }

  const effectivePreferredGpuType =
    (setup.failoverCount ?? 0) >= 2 ? undefined : preferredGpuType;

  const canReuseExistingPlacement =
    Boolean(setup.runpodVolumeId && setup.runpodVolumeDatacenterId) &&
    Boolean(setup.runpodVolumeDatacenterId && SERVERLESS_DATACENTER_IDS.has(setup.runpodVolumeDatacenterId)) &&
    (!effectivePreferredGpuType ||
      !setup.runpodGpuType ||
      setup.runpodGpuType === effectivePreferredGpuType) &&
    (!setup.runpodGpuType ||
      !setup.runpodVolumeDatacenterId ||
      placementKey(setup.runpodVolumeDatacenterId, setup.runpodGpuType) !== setup.lastFailedPlacementKey);

  let selectedPlacement: PlacementCandidate | null = null;
  if (canReuseExistingPlacement && setup.runpodVolumeDatacenterId) {
    try {
      selectedPlacement = await bestPlacementForDatacenter(
        setup.runpodVolumeDatacenterId,
        minVramGb,
        effectivePreferredGpuType
      );
    } catch {
      selectedPlacement = null;
    }
  }

  if (!selectedPlacement) {
    const bestByDatacenter = new Map<string, PlacementCandidate>();
    const preferenceList = buildDatacenterPreferenceList(env.RUNPOD_DATACENTER_ID).slice(0, AUTOPICK_MAX_DATACENTERS);

    for (const datacenterId of preferenceList) {
      try {
        const best = await bestPlacementForDatacenter(
          datacenterId,
          minVramGb,
          effectivePreferredGpuType
        );
        if (best) {
          bestByDatacenter.set(datacenterId, best);
        }
      } catch {
        // Ignore transient RunPod market errors; continue scanning other datacenters.
      }
    }

    if (bestByDatacenter.size === 0) {
      try {
        const allDatacenters = await listDataCenterIds();
        const preferredRegion = env.RUNPOD_DATACENTER_ID.split("-")[0] ?? "US";
        const regionDatacenters = allDatacenters
          .filter((datacenterId) => SERVERLESS_DATACENTER_IDS.has(datacenterId))
          .filter((datacenterId) => datacenterId.startsWith(`${preferredRegion}-`))
          .slice(0, AUTOPICK_MAX_DATACENTERS);

        for (const datacenterId of regionDatacenters) {
          if (bestByDatacenter.has(datacenterId)) {
            continue;
          }
          try {
            const best = await bestPlacementForDatacenter(
              datacenterId,
              minVramGb,
              effectivePreferredGpuType
            );
            if (best) {
              bestByDatacenter.set(datacenterId, best);
            }
          } catch {
            // Ignore and continue.
          }
        }
      } catch {
        // Ignore and fall through to the final empty-candidate error.
      }
    }

    const placementCandidates = [...bestByDatacenter.values()].sort((a, b) =>
      comparePlacementCandidates(a, b, effectivePreferredGpuType)
    );
    const filteredPlacementCandidates = setup.lastFailedPlacementKey
      ? placementCandidates.filter(
          (candidate) =>
            placementKey(candidate.datacenterId, candidate.gpuType) !== setup.lastFailedPlacementKey
        )
      : placementCandidates;
    const orderedPlacementCandidates =
      filteredPlacementCandidates.length > 0 ? filteredPlacementCandidates : placementCandidates;
    if (orderedPlacementCandidates.length === 0) {
      throw new Error(`RunPod autopick failed: no CUDA GPUs with >=${minVramGb}GB found.`);
    }

    const volumeProvisionErrors: string[] = [];
    for (const candidate of orderedPlacementCandidates) {
      const candidateVolumeName = `carcompose-models-${normalizeDatacenterToken(candidate.datacenterId)}`;
      try {
        const volumeId = await ensureVolume({
          name: candidateVolumeName,
          sizeGb: env.RUNPOD_VOLUME_GB,
          datacenterId: candidate.datacenterId
        });

        setup = await putSetupState({
          runpodVolumeId: volumeId,
          runpodVolumeDatacenterId: candidate.datacenterId,
          runpodGpuType: candidate.gpuType,
          lastFailedPlacementKey: ""
        });
        selectedPlacement = candidate;
        break;
      } catch (error) {
        const message = errorMessage(error);
        const lowered = message.toLowerCase();
        const isDatacenterError =
          isDatacenterNotFoundError(error) ||
          lowered.includes("storage clusters available") ||
          lowered.includes("no storage clusters");
        if (isDatacenterError) {
          volumeProvisionErrors.push(`${candidate.datacenterId}: ${message}`);
          continue;
        }
        throw new Error(`RunPod volume provisioning failed for datacenter "${candidate.datacenterId}". ${message}`);
      }
    }

    if (!selectedPlacement || !setup.runpodVolumeId || !setup.runpodVolumeDatacenterId) {
      const detail = volumeProvisionErrors.length ? ` Attempts: ${volumeProvisionErrors.join(" | ")}` : "";
      throw new Error(`RunPod autopick failed: unable to provision a network volume.${detail}`);
    }
  }

  const activeVolumeDatacenterId =
    canReuseExistingPlacement && setup.runpodVolumeDatacenterId
      ? setup.runpodVolumeDatacenterId
      : selectedPlacement.datacenterId;
  const selectedGpuType =
    canReuseExistingPlacement && setup.runpodGpuType
      ? setup.runpodGpuType
      : selectedPlacement.gpuType;
  const volumeName = `carcompose-models-${normalizeDatacenterToken(activeVolumeDatacenterId)}`;

  const provisioningHash = computeProvisioningHash({
    workerImage: resolvedWorkerImage,
    templateEnv,
    datacenterId: activeVolumeDatacenterId,
    volumeName,
    volumeGb: env.RUNPOD_VOLUME_GB,
    requestedGpuType: preferredGpuType,
    gpuType: selectedGpuType,
    workersMin: env.RUNPOD_WORKERS_MIN,
    workersMax: env.RUNPOD_WORKERS_MAX,
    idleTimeout: env.RUNPOD_IDLE_TIMEOUT_S,
    executionTimeoutS: env.RUNPOD_EXECUTION_TIMEOUT_S,
    registryAuthName: registryAuthName ?? null
  });
  const desiredTemplateName = `carcompose-worker-template-${provisioningHash}`;
  const desiredEndpointName = `carcompose-pipeline-${provisioningHash}`;

  if (
    setup.bucketName !== env.R2_BUCKET_NAME ||
    setup.workerImage !== resolvedWorkerImage ||
    setup.workerImageDigest !== resolvedWorkerImageDigest ||
    setup.provisioningHash !== provisioningHash
  ) {
    setup = await putSetupState({
      bucketName: env.R2_BUCKET_NAME,
      workerImage: resolvedWorkerImage,
      workerImageDigest: resolvedWorkerImageDigest,
      provisioningHash,
      runpodTemplateId: "",
      runpodEndpointId: "",
      initJobId: "",
      initJobStatus: "NOT_STARTED",
      runpodGpuType: selectedGpuType
    });
  } else if (setup.runpodGpuType !== selectedGpuType) {
    setup = await putSetupState({ runpodGpuType: selectedGpuType });
  }

  await cleanupStaleCarcomposeEndpoints({
    keepEndpointId: setup.runpodEndpointId || undefined,
    keepEndpointName: desiredEndpointName
  });

  if (!setup.runpodTemplateId) {
    let templateId = "";
    try {
      templateId = await ensureTemplate({
        existingId: setup.runpodTemplateId,
        name: desiredTemplateName,
        dockerImage: resolvedWorkerImage,
        volumeMountPath: "/runpod-volume",
        env: templateEnv,
        registryAuthId
      });
    } catch (error) {
      throw new Error(`RunPod template provisioning failed: ${errorMessage(error)}`);
    }

    setup = await putSetupState({ runpodTemplateId: templateId });
  }

  if (!setup.runpodEndpointId) {
    if (!setup.runpodTemplateId || !setup.runpodVolumeId) {
      throw new Error("Provisioning error: missing template or volume ID.");
    }

    let endpointId = "";
    try {
      endpointId = await ensureEndpoint({
        existingId: setup.runpodEndpointId,
        name: desiredEndpointName,
        templateId: setup.runpodTemplateId,
        volumeId: setup.runpodVolumeId,
        gpuType: selectedGpuType,
        workersMin: env.RUNPOD_WORKERS_MIN,
        workersMax: env.RUNPOD_WORKERS_MAX,
        idleTimeout: env.RUNPOD_IDLE_TIMEOUT_S,
        executionTimeoutMs: env.RUNPOD_EXECUTION_TIMEOUT_S * 1000
      });
    } catch (error) {
      throw new Error(
        `RunPod endpoint provisioning failed for RUNPOD_GPU_TYPE="${selectedGpuType}". ${errorMessage(error)}`
      );
    }

    setup = await putSetupState({ runpodEndpointId: endpointId });
  }

  if (!setup.runpodEndpointId) {
    throw new Error("Provisioning error: missing endpoint ID.");
  }
  const endpointId = setup.runpodEndpointId;

  try {
    await patchRunpodEndpointRest(endpointId, {
      dataCenterIds: [activeVolumeDatacenterId],
      gpuTypeIds: [selectedGpuType],
      workersMin: env.RUNPOD_WORKERS_MIN,
      workersMax: env.RUNPOD_WORKERS_MAX,
      idleTimeout: env.RUNPOD_IDLE_TIMEOUT_S,
      executionTimeoutMs: env.RUNPOD_EXECUTION_TIMEOUT_S * 1000
    });
  } catch (error) {
    throw new Error(`RunPod endpoint update failed: ${errorMessage(error)}`);
  }

  if (!setup.initJobId || setup.initJobStatus === "FAILED") {
    const initJobId = await submitRunpodJob(endpointId, {
      action: "download_models"
    });
    setup = await putSetupState({ initJobId, initJobStatus: "RUNNING" });

    return {
      ready: false,
      phase: "downloading_models",
      message: "Model download job started.",
      details: {
        bucket: env.R2_BUCKET_NAME,
        workerImage: resolvedWorkerImage,
        workerImageDigest: resolvedWorkerImageDigest,
        volumeDatacenterId: activeVolumeDatacenterId,
        endpointId,
        initJobId
      }
    };
  }

  const initStatus = await getRunpodJobStatus(endpointId, setup.initJobId);
  let mapped = mapInitStatus(initStatus.status);

  // Defensive: if the worker returns `{status:"error"}` but RunPod reports COMPLETED,
  // treat init as failed. Readiness must never go green on an errored init output.
  if (mapped === "COMPLETED") {
    const workerError = readWorkerErrorMessage(initStatus.output);
    if (workerError) {
      mapped = "FAILED";
      setup = await putSetupState({ initJobStatus: mapped });
      return {
        ready: false,
        phase: "error",
        message: workerError,
        details: {
          bucket: env.R2_BUCKET_NAME,
          workerImage: resolvedWorkerImage,
          workerImageDigest: resolvedWorkerImageDigest,
          volumeDatacenterId: activeVolumeDatacenterId,
          endpointId,
          initJobId: setup.initJobId
        }
      };
    }
  }

  setup = await putSetupState(
    mapped === "COMPLETED"
      ? { initJobStatus: mapped, failoverCount: 0, lastFailedPlacementKey: "" }
      : { initJobStatus: mapped }
  );

  if (mapped === "COMPLETED") {
    return {
      ready: true,
      phase: "ready",
      message: "System is ready.",
      details: {
        bucket: env.R2_BUCKET_NAME,
        workerImage: resolvedWorkerImage,
        workerImageDigest: resolvedWorkerImageDigest,
        volumeDatacenterId: activeVolumeDatacenterId,
        endpointId,
        initJobId: setup.initJobId
      }
    };
  }

  if (mapped === "FAILED") {
    const jobErrorMessage =
      readWorkerErrorMessage(initStatus.output) ?? initStatus.error ?? "Model initialization failed.";
    return {
      ready: false,
      phase: "error",
      message: jobErrorMessage,
      details: {
        bucket: env.R2_BUCKET_NAME,
        workerImage: resolvedWorkerImage,
        workerImageDigest: resolvedWorkerImageDigest,
        endpointId: setup.runpodEndpointId,
        initJobId: setup.initJobId
      }
    };
  }

  if (initStatus.status === "IN_QUEUE") {
    try {
      const health = await getRunpodHealth(endpointId);
      const workers = health.workers ?? {};
      const runningLikeCount =
        (workers.running ?? 0) + (workers.ready ?? 0) + (workers.initializing ?? 0);

      if (runningLikeCount === 0) {
        let delayTimeMs = 0;
        try {
          const requests = await getRunpodRequests(endpointId);
          const match = requests.requests?.find((item) => item.id === setup.initJobId);
          delayTimeMs = typeof match?.delayTime === "number" ? match.delayTime : 0;
        } catch {
          // Ignore request probe failures.
        }

        const nowMs = Date.now();
        const lastFailoverMs = setup.lastFailoverAt ? Date.parse(setup.lastFailoverAt) : 0;
        const failoverCooldownOk =
          !Number.isFinite(lastFailoverMs) || nowMs - lastFailoverMs >= AUTOPICK_QUEUE_FAILOVER_COOLDOWN_MS;
        const failoverThresholdHit = delayTimeMs >= AUTOPICK_QUEUE_FAILOVER_AFTER_MS;

        if (failoverThresholdHit && failoverCooldownOk) {
          // Best-effort: stop burning queue time on a dead placement.
          try {
            await purgeRunpodQueue(endpointId);
          } catch {
            // Ignore.
          }
          try {
            await patchRunpodEndpointRest(endpointId, { workersMin: 0, workersMax: 0 });
          } catch {
            // Ignore.
          }
          try {
            await deleteRunpodEndpointRest(endpointId);
          } catch {
            // Ignore.
          }

          const volumeId = setup.runpodVolumeId;
          if (volumeId) {
            try {
              await deleteNetworkVolume(volumeId);
            } catch {
              // Ignore volume deletion failures; stale volume cleanup can be manual.
            }
          }

          setup = await putSetupState({
            provisioningHash: "",
            runpodVolumeId: "",
            runpodVolumeDatacenterId: "",
            runpodTemplateId: "",
            runpodEndpointId: "",
            runpodGpuType: "",
            lastFailedPlacementKey: placementKey(activeVolumeDatacenterId, selectedGpuType),
            initJobId: "",
            initJobStatus: "NOT_STARTED",
            lastFailoverAt: new Date().toISOString(),
            failoverCount: (setup.failoverCount ?? 0) + 1
          });

          return {
            ready: false,
            phase: "provisioning",
            message:
              "Model init job has been queued too long with no workers active. " +
              "Cleared the queue and will reprovision in a different location/GPU on the next poll.",
            details: {
              bucket: env.R2_BUCKET_NAME,
              workerImage: resolvedWorkerImage,
              workerImageDigest: resolvedWorkerImageDigest,
              volumeDatacenterId: activeVolumeDatacenterId
            }
          };
        }

        return {
          ready: false,
          phase: "downloading_models",
          message:
            "Model init job is queued waiting for GPU capacity. " +
            `No workers are active for ${selectedGpuType} in ${activeVolumeDatacenterId} right now.`,
          details: {
            bucket: env.R2_BUCKET_NAME,
            workerImage: resolvedWorkerImage,
            workerImageDigest: resolvedWorkerImageDigest,
            volumeDatacenterId: activeVolumeDatacenterId,
            endpointId,
            initJobId: setup.initJobId
          }
        };
      }
    } catch {
      // Keep readiness route resilient even if health probe fails.
    }
  }

  return {
    ready: false,
    phase: "downloading_models",
    message: "Model download in progress.",
    details: {
      bucket: env.R2_BUCKET_NAME,
      workerImage: resolvedWorkerImage,
      workerImageDigest: resolvedWorkerImageDigest,
      volumeDatacenterId: activeVolumeDatacenterId,
      endpointId,
      initJobId: setup.initJobId
    }
  };
}

export async function getProvisionedSetupOrThrow(): Promise<SetupState> {
  const setup = await getSetupState();
  if (!setup || !setup.runpodEndpointId || setup.initJobStatus !== "COMPLETED") {
    throw new Error("System is initializing. Try again soon.");
  }

  return setup;
}
