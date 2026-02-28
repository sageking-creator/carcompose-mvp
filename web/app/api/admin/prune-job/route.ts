import { NextResponse } from "next/server";
import { z } from "zod";
import { assertPasscode } from "@/lib/auth";
import { getEnv } from "@/lib/env";
import { jsonError } from "@/lib/http";
import { deleteKeys, listKeysByPrefix, objectExists } from "@/lib/r2";

const requestSchema = z.object({
  jobId: z.string().uuid(),
  targets: z
    .array(z.enum(["debug", "uploads", "masks", "outputs", "jobs"]))
    .min(1)
    .optional()
});

const DEFAULT_TARGETS = ["debug", "uploads", "masks", "jobs"] as const;

export async function POST(request: Request): Promise<NextResponse> {
  try {
    const env = getEnv();
    const unauthorized = assertPasscode(request, env.APP_PASSCODE);
    if (unauthorized) {
      return unauthorized;
    }

    const payload = requestSchema.parse(await request.json());
    const targets = payload.targets ?? [...DEFAULT_TARGETS];
    const keys = new Set<string>();

    const prefixTargets: Record<string, string> = {
      debug: `debug/${payload.jobId}/`,
      uploads: `uploads/${payload.jobId}/`,
      masks: `masks/${payload.jobId}/`,
      outputs: `outputs/${payload.jobId}/`
    };

    for (const target of targets) {
      const prefix = prefixTargets[target];
      if (!prefix) {
        continue;
      }
      const matched = await listKeysByPrefix(prefix);
      for (const key of matched) {
        keys.add(key);
      }
    }

    if (targets.includes("jobs")) {
      const jobStateKey = `jobs/${payload.jobId}.json`;
      if (await objectExists(jobStateKey)) {
        keys.add(jobStateKey);
      }
    }

    const matchedKeys = Array.from(keys).sort();
    const deleted = await deleteKeys(matchedKeys);

    return NextResponse.json({
      ok: true,
      jobId: payload.jobId,
      targets,
      matched: matchedKeys.length,
      deleted
    });
  } catch (error) {
    return jsonError(error);
  }
}
