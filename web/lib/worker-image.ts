import { type AppEnv } from "@/lib/env";

export function resolveWorkerImage(env: AppEnv): { image: string | null; reason?: string } {
  if (env.WORKER_IMAGE) {
    return { image: env.WORKER_IMAGE };
  }

  if (env.VERCEL_GIT_REPO_OWNER && env.VERCEL_GIT_REPO_SLUG) {
    const owner = env.VERCEL_GIT_REPO_OWNER.toLowerCase();
    const slug = env.VERCEL_GIT_REPO_SLUG.toLowerCase();
    return { image: `ghcr.io/${owner}/${slug}-worker:main` };
  }

  return {
    image: null,
    reason: "Unable to infer worker image. Set WORKER_IMAGE or deploy from connected Vercel Git repo."
  };
}
