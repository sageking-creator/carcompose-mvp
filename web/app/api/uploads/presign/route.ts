import { randomUUID } from "node:crypto";
import { NextResponse } from "next/server";
import { z } from "zod";
import { assertPasscode } from "@/lib/auth";
import { ensureBucketExists } from "@/lib/cloudflare-r2-admin";
import { getEnv } from "@/lib/env";
import { jsonError } from "@/lib/http";
import { ensureCorsRules, presignPut } from "@/lib/r2";
import {
  buildUploadKeys,
  normalizeContentType,
  validateUploadContentTypes
} from "@/lib/uploads";

const requestSchema = z.object({
  contentTypes: z.object({
    car: z.string(),
    background: z.string()
  })
});

export async function POST(request: Request): Promise<NextResponse> {
  try {
    const env = getEnv();
    const unauthorized = assertPasscode(request, env.APP_PASSCODE);
    if (unauthorized) {
      return unauthorized;
    }

    await ensureBucketExists(env.R2_BUCKET_NAME);
    await ensureCorsRules(env.R2_BUCKET_NAME);

    const payload = requestSchema.parse(await request.json());
    validateUploadContentTypes(payload.contentTypes);

    const jobId = randomUUID();
    const { carKey, backgroundKey } = buildUploadKeys(jobId);

    const [carPutUrl, backgroundPutUrl] = await Promise.all([
      presignPut(carKey, normalizeContentType(payload.contentTypes.car), 3600),
      presignPut(backgroundKey, normalizeContentType(payload.contentTypes.background), 3600)
    ]);

    return NextResponse.json({
      jobId,
      car: { key: carKey, putUrl: carPutUrl },
      background: { key: backgroundKey, putUrl: backgroundPutUrl }
    });
  } catch (error) {
    return jsonError(error, 400);
  }
}
