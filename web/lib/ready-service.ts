import { ensureBucketExists } from "@/lib/cloudflare-r2-admin";
import { type AppEnv } from "@/lib/env";
import { resolveGhcrImageToDigest } from "@/lib/ghcr";
import { ensureLifecycleRules } from "@/lib/r2";
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
    volumeDatacenterId?: string;
    endpointId?: string;
    initJobId?: string;
  };
};

const VOLUME_DATACENTER_FALLBACKS = ["US-NC-1", "US-TX-3"];

const AUTOPICK_SECURE_CLOUD = true;
const AUTOPICK_MAX_DATACENTERS = 25;
const AUTOPICK_QUEUE_FAILOVER_AFTER_MS = 5 * 60 * 1000;
const AUTOPICK_QUEUE_FAILOVER_COOLDOWN_MS = 10 * 60 * 1000;
const VOLUME_DATACENTER_CANDIDATES = [
  // Known-working (network volume create/delete verified in this account in the past).
  "US-MD-1",
  "US-TX-3",
  "US-MO-2",
  "US-GA-2",
  "US-CA-2",
  "US-KS-2",
  "US-NC-1"
];

function getTemplateEnv(env: AppEnv): Array<{ key: string; value: string }> {
  const cacheDir = env.MODEL_CACHE_DIR;
  const hfHome = env.HF_HOME;

  return [
    { key: "MODEL_CACHE_DIR", value: cacheDir },
    { key: "HF_HOME", value: hfHome },
    { key: "TRANSFORMERS_CACHE", value: hfHome },
    { key: "LIBCOM_MODEL_DIR", value: `${cacheDir}/libcom` },
    { key: "CONTROLCOM_CKPT", value: `${cacheDir}/controlcom/ControlCom_blend_harm.pth` },
    { key: "CLIP_MODEL_DIR", value: `${cacheDir}/controlcom/openai-clip-vit-large-patch14` },
    { key: "CUDA_VISIBLE_DEVICES", value: "0" },
    { key: "PYTORCH_CUDA_ALLOC_CONF", value: "max_split_size_mb:512" },
    { key: "PIPELINE_VARIANT", value: env.PIPELINE_VARIANT }
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

function comparePlacementCandidates(a: PlacementCandidate, b: PlacementCandidate): number {
  const stockDelta = stockRank(a.stockStatus) - stockRank(b.stockStatus);
  if (stockDelta !== 0) {
    return stockDelta;
  }

  const priceDelta = a.pricePerHour - b.pricePerHour;
  if (priceDelta !== 0) {
    return priceDelta;
  }

  const maxDelta = b.maxUnreservedGpuCount - a.maxUnreservedGpuCount;
  if (maxDelta !== 0) {
    return maxDelta;
  }

  return b.memoryInGb - a.memoryInGb;
}

async function bestPlacementForDatacenter(
  datacenterId: string,
  minVramGb: number
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

  candidates.sort(comparePlacementCandidates);
  return candidates[0] ?? null;
}

function buildDatacenterPreferenceList(preferredDatacenterId: string, extra: string[] = []): string[] {
  const seen = new Set<string>();
  const ordered = [...extra, preferredDatacenterId, ...VOLUME_DATACENTER_CANDIDATES, ...VOLUME_DATACENTER_FALLBACKS]
    .filter(Boolean)
    .filter((dc) => {
      if (seen.has(dc)) {
        return false;
      }
      seen.add(dc);
      return true;
    });
  return ordered;
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
    try {
      const health = await getRunpodHealth(endpoint.id);
      const workers = health.workers ?? {};
      const jobs = health.jobs ?? {};
      const runningLikeCount =
        (workers.running ?? 0) + (workers.ready ?? 0) + (workers.initializing ?? 0);
      const inProgressCount = jobs.inProgress ?? 0;
      if (runningLikeCount > 0 || inProgressCount > 0) {
        continue;
      }
    } catch {
      // If health fails, attempt deletion anyway (purge/delete are best-effort).
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
      details: { bucket: env.R2_BUCKET_NAME }
    };
  }

  const registryAuth =
    env.GHCR_USERNAME && env.GHCR_TOKEN
      ? { username: env.GHCR_USERNAME, password: env.GHCR_TOKEN }
      : undefined;

  const resolvedWorkerImage = await resolveGhcrImageToDigest(workerImage.image, registryAuth);
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
  await ensureLifecycleRules(env.R2_BUCKET_NAME);

  const templateEnv = getTemplateEnv(env)
    .slice()
    .sort((a, b) => a.key.localeCompare(b.key));
  const minVramGb = minVramGbForVariant(env.PIPELINE_VARIANT);

  let setup: SetupState =
    (await getSetupState()) ??
    (await putSetupState({
      bucketName: env.R2_BUCKET_NAME,
      workerImage: resolvedWorkerImage,
      initJobStatus: "NOT_STARTED"
    }));

  if (setup.bucketName !== env.R2_BUCKET_NAME || setup.workerImage !== resolvedWorkerImage) {
    setup = await putSetupState({
      bucketName: env.R2_BUCKET_NAME,
      workerImage: resolvedWorkerImage
    });
  }

  let selectedPlacement: PlacementCandidate | null = null;
  if (setup.runpodVolumeId && setup.runpodVolumeDatacenterId) {
    try {
      selectedPlacement = await bestPlacementForDatacenter(setup.runpodVolumeDatacenterId, minVramGb);
    } catch {
      selectedPlacement = null;
    }
  }

  if (!selectedPlacement) {
    const bestByDatacenter = new Map<string, PlacementCandidate>();
    const preferenceList = buildDatacenterPreferenceList(env.RUNPOD_DATACENTER_ID).slice(0, AUTOPICK_MAX_DATACENTERS);

    for (const datacenterId of preferenceList) {
      try {
        const best = await bestPlacementForDatacenter(datacenterId, minVramGb);
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
          .filter((datacenterId) => datacenterId.startsWith(`${preferredRegion}-`))
          .slice(0, AUTOPICK_MAX_DATACENTERS);

        for (const datacenterId of regionDatacenters) {
          if (bestByDatacenter.has(datacenterId)) {
            continue;
          }
          try {
            const best = await bestPlacementForDatacenter(datacenterId, minVramGb);
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

    const placementCandidates = [...bestByDatacenter.values()].sort(comparePlacementCandidates);
    if (placementCandidates.length === 0) {
      throw new Error(`RunPod autopick failed: no CUDA GPUs with >=${minVramGb}GB found.`);
    }

    const volumeProvisionErrors: string[] = [];
    for (const candidate of placementCandidates) {
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
          runpodGpuType: candidate.gpuType
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

  const activeVolumeDatacenterId = setup.runpodVolumeDatacenterId ?? selectedPlacement.datacenterId;
  const selectedGpuType = setup.runpodGpuType ?? selectedPlacement.gpuType;
  const volumeName = `carcompose-models-${normalizeDatacenterToken(activeVolumeDatacenterId)}`;

  const provisioningHash = computeProvisioningHash({
    workerImage: resolvedWorkerImage,
    templateEnv,
    datacenterId: activeVolumeDatacenterId,
    volumeName,
    volumeGb: env.RUNPOD_VOLUME_GB,
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
    setup.provisioningHash !== provisioningHash
  ) {
    setup = await putSetupState({
      bucketName: env.R2_BUCKET_NAME,
      workerImage: resolvedWorkerImage,
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
          volumeDatacenterId: activeVolumeDatacenterId,
          endpointId,
          initJobId: setup.initJobId
        }
      };
    }
  }

  setup = await putSetupState({ initJobStatus: mapped });

  if (mapped === "COMPLETED") {
    return {
      ready: true,
      phase: "ready",
      message: "System is ready.",
      details: {
        bucket: env.R2_BUCKET_NAME,
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
