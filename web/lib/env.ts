import { z } from "zod";
import { EnvError } from "@/lib/errors";

const envSchema = z.object({
  APP_PASSCODE: z
    .string({ required_error: "APP_PASSCODE is required" })
    .min(1, "APP_PASSCODE is required"),
  RUNPOD_API_KEY: z
    .string({ required_error: "RUNPOD_API_KEY is required" })
    .min(1, "RUNPOD_API_KEY is required"),
  CLOUDFLARE_ACCOUNT_ID: z
    .string({ required_error: "CLOUDFLARE_ACCOUNT_ID is required" })
    .min(1, "CLOUDFLARE_ACCOUNT_ID is required"),
  CLOUDFLARE_API_TOKEN: z
    .string({ required_error: "CLOUDFLARE_API_TOKEN is required" })
    .min(1, "CLOUDFLARE_API_TOKEN is required"),
  R2_ACCESS_KEY_ID: z
    .string({ required_error: "R2_ACCESS_KEY_ID is required" })
    .min(1, "R2_ACCESS_KEY_ID is required"),
  R2_SECRET_ACCESS_KEY: z
    .string({ required_error: "R2_SECRET_ACCESS_KEY is required" })
    .min(1, "R2_SECRET_ACCESS_KEY is required"),
  R2_BUCKET_NAME: z.string().min(1).default("carcompose-storage"),
  R2_ENDPOINT_URL: z
    .string({ required_error: "R2_ENDPOINT_URL is required" })
    .min(1, "R2_ENDPOINT_URL is required")
    .url("R2_ENDPOINT_URL must be a valid URL"),
  RUNPOD_DATACENTER_ID: z.string().default("US-TX-3"),
  RUNPOD_GPU_TYPE: z.string().default("NVIDIA GeForce RTX 4090"),
  RUNPOD_VOLUME_GB: z.coerce.number().int().positive().default(50),
  RUNPOD_WORKERS_MIN: z.coerce.number().int().nonnegative().default(0),
  RUNPOD_WORKERS_MAX: z.coerce.number().int().positive().default(3),
  RUNPOD_IDLE_TIMEOUT_S: z.coerce.number().int().positive().default(60),
  RUNPOD_EXECUTION_TIMEOUT_S: z.coerce.number().int().positive().default(3600),
  PIPELINE_VARIANT: z.enum(["core", "full"]).default("core"),
  MODEL_CACHE_DIR: z.string().default("/runpod-volume/models"),
  HF_HOME: z.string().default("/runpod-volume/hf_cache"),
  BIREFNET_REPO_ID: z.string().default("ZhengPeng7/BiRefNet_dynamic-matting"),
  BIREFNET_INFER_RES: z.coerce.number().int().refine((value) => value === 1024 || value === 2048, {
    message: "BIREFNET_INFER_RES must be 1024 or 2048"
  }).default(2048),
  MAX_OUTPUT_LONG_EDGE: z.coerce.number().int().positive().default(2048),
  OUTPUT_RESIZE_MODE: z.enum(["preserve", "stretch"]).default("preserve"),
  CORE_CONTACT_SHADOW_STRENGTH: z.coerce.number().min(0).max(1).default(0.32),
  CONTACT_SHADOW_MODE: z.enum(["v1", "v2"]).default("v2"),
  GLASS_NORMALIZATION_MODE: z.enum(["off", "auto", "force"]).default("off"),
  STUDIO_MODE: z.enum(["off", "auto", "on"]).default("auto"),
  STUDIO_CAR_WIDTH_RATIO: z.coerce.number().min(0.4).max(0.95).default(0.82),
  STUDIO_GROUND_RATIO: z.coerce.number().min(0.6).max(0.98).default(0.9),
  MAX_EDGE_HALO_MEAN_DELTA: z.coerce.number().positive().default(14),
  MAX_EDGE_BAND_WIDTH_PX: z.coerce.number().positive().default(7.5),
  DEBUG_ARTIFACTS: z
    .string()
    .default("false")
    .transform((value) => ["1", "true", "yes", "on"].includes(value.toLowerCase())),
  WORKER_IMAGE: z.string().optional(),
  GHCR_USERNAME: z.string().optional(),
  GHCR_TOKEN: z.string().optional(),
  VERCEL_GIT_REPO_OWNER: z.string().optional(),
  VERCEL_GIT_REPO_SLUG: z.string().optional(),
  VERCEL_GIT_COMMIT_SHA: z.string().optional()
});

export type AppEnv = z.infer<typeof envSchema>;

let cachedEnv: AppEnv | null = null;

export function getEnv(): AppEnv {
  if (cachedEnv) {
    return cachedEnv;
  }

  const parsed = envSchema.safeParse(process.env);
  if (!parsed.success) {
    const message = parsed.error.issues[0]?.message ?? "Invalid environment";
    throw new EnvError(message);
  }

  cachedEnv = parsed.data;
  return cachedEnv;
}

export function resetEnvCacheForTests(): void {
  cachedEnv = null;
}
