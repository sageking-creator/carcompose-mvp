import { type AppEnv } from "@/lib/env";

function getShortCommitSha(value: string | undefined): string | null {
  if (!value) {
    return null;
  }
  const normalized = value.trim().toLowerCase();
  if (!/^[0-9a-f]{7,40}$/.test(normalized)) {
    return null;
  }
  return normalized.slice(0, 7);
}

export function resolveWorkerImage(env: AppEnv): { image: string | null; reason?: string } {
  if (env.WORKER_IMAGE) {
    return { image: env.WORKER_IMAGE };
  }

  if (env.VERCEL_GIT_REPO_OWNER && env.VERCEL_GIT_REPO_SLUG) {
    const owner = env.VERCEL_GIT_REPO_OWNER.toLowerCase();
    const slug = env.VERCEL_GIT_REPO_SLUG.toLowerCase();
    const commitSha = getShortCommitSha(env.VERCEL_GIT_COMMIT_SHA);
    if (commitSha) {
      return { image: `ghcr.io/${owner}/${slug}-worker:sha-${commitSha}` };
    }
    return { image: `ghcr.io/${owner}/${slug}-worker:main` };
  }

  return {
    image: null,
    reason: "Unable to infer worker image. Set WORKER_IMAGE or deploy from connected Vercel Git repo."
  };
}
