import { ensureBucketExists } from "@/lib/cloudflare-r2-admin";
import { type AppEnv } from "@/lib/env";
import { resolveGhcrImageToDigest } from "@/lib/ghcr";
import { ensureLifecycleRules } from "@/lib/r2";
import {
  ensureEndpoint,
  ensureRegistryAuth,
  ensureTemplate,
  ensureVolume
} from "@/lib/runpod-admin";
import { getRunpodHealth, getRunpodJobStatus, submitRunpodJob } from "@/lib/runpod-jobs";
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
    endpointId?: string;
    initJobId?: string;
  };
};

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

function normalizeRegistryAuthName(username: string): string {
  const cleaned = username.toLowerCase().replace(/[^a-z0-9-]+/g, "-").replace(/-+/g, "-");
  const trimmed = cleaned.replace(/^-+/, "").replace(/-+$/, "");
  return `carcompose-ghcr-${trimmed || "user"}`.slice(0, 63);
}

function normalizeDatacenterToken(datacenterId: string): string {
  return datacenterId.toLowerCase().replace(/[^a-z0-9-]+/g, "-");
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
  const volumeName = `carcompose-models-${normalizeDatacenterToken(env.RUNPOD_DATACENTER_ID)}`;
  const provisioningHash = computeProvisioningHash({
    workerImage: resolvedWorkerImage,
    templateEnv,
    datacenterId: env.RUNPOD_DATACENTER_ID,
    volumeName,
    volumeGb: env.RUNPOD_VOLUME_GB,
    gpuType: env.RUNPOD_GPU_TYPE,
    workersMin: env.RUNPOD_WORKERS_MIN,
    workersMax: env.RUNPOD_WORKERS_MAX,
    idleTimeout: env.RUNPOD_IDLE_TIMEOUT_S,
    executionTimeoutS: env.RUNPOD_EXECUTION_TIMEOUT_S,
    registryAuthName: registryAuthName ?? null
  });
  const desiredTemplateName = `carcompose-worker-template-${provisioningHash}`;
  const desiredEndpointName = `carcompose-pipeline-${provisioningHash}`;

  let setup = await getSetupState();
  if (!setup) {
    setup = await putSetupState({
      bucketName: env.R2_BUCKET_NAME,
      workerImage: resolvedWorkerImage,
      provisioningHash,
      initJobStatus: "NOT_STARTED"
    });
  } else if (
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
      initJobStatus: "NOT_STARTED"
    });
  }

  if (!setup.runpodVolumeId) {
    let volumeId = "";
    try {
      volumeId = await ensureVolume({
        existingId: setup.runpodVolumeId,
        name: volumeName,
        sizeGb: env.RUNPOD_VOLUME_GB,
        datacenterId: env.RUNPOD_DATACENTER_ID
      });
    } catch (error) {
      throw new Error(
        `RunPod volume provisioning failed for RUNPOD_DATACENTER_ID="${env.RUNPOD_DATACENTER_ID}". ${errorMessage(error)}`
      );
    }

    setup = await putSetupState({ runpodVolumeId: volumeId });
  }

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
        gpuType: env.RUNPOD_GPU_TYPE,
        workersMin: env.RUNPOD_WORKERS_MIN,
        workersMax: env.RUNPOD_WORKERS_MAX,
        idleTimeout: env.RUNPOD_IDLE_TIMEOUT_S,
        executionTimeoutMs: env.RUNPOD_EXECUTION_TIMEOUT_S * 1000
      });
    } catch (error) {
      throw new Error(
        `RunPod endpoint provisioning failed for RUNPOD_GPU_TYPE="${env.RUNPOD_GPU_TYPE}". ${errorMessage(error)}`
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
        return {
          ready: false,
          phase: "downloading_models",
          message:
            "Model init job is queued waiting for GPU capacity. " +
            `No workers are active for ${env.RUNPOD_GPU_TYPE} in ${env.RUNPOD_DATACENTER_ID} right now.`,
          details: {
            bucket: env.R2_BUCKET_NAME,
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
