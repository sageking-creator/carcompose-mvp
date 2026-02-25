import { z } from "zod";
import { EnvError } from "@/lib/errors";

const envSchema = z.object({
  APP_PASSCODE: z.string().min(1, "APP_PASSCODE is required"),
  RUNPOD_API_KEY: z.string().min(1, "RUNPOD_API_KEY is required"),
  CLOUDFLARE_ACCOUNT_ID: z.string().min(1, "CLOUDFLARE_ACCOUNT_ID is required"),
  CLOUDFLARE_API_TOKEN: z.string().min(1, "CLOUDFLARE_API_TOKEN is required"),
  R2_ACCESS_KEY_ID: z.string().min(1, "R2_ACCESS_KEY_ID is required"),
  R2_SECRET_ACCESS_KEY: z.string().min(1, "R2_SECRET_ACCESS_KEY is required"),
  R2_BUCKET_NAME: z.string().min(1).default("carcompose-storage"),
  R2_ENDPOINT_URL: z.string().url("R2_ENDPOINT_URL must be a valid URL"),
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
  WORKER_IMAGE: z.string().optional(),
  GHCR_USERNAME: z.string().optional(),
  GHCR_TOKEN: z.string().optional(),
  VERCEL_GIT_REPO_OWNER: z.string().optional(),
  VERCEL_GIT_REPO_SLUG: z.string().optional()
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
