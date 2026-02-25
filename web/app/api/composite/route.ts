import { NextResponse } from "next/server";
import { z } from "zod";
import { assertPasscode } from "@/lib/auth";
import { getEnv } from "@/lib/env";
import { jsonError } from "@/lib/http";
import { getProvisionedSetupOrThrow } from "@/lib/ready-service";
import { presignGet, presignPut } from "@/lib/r2";
import { putJobState } from "@/lib/r2-state";
import { submitRunpodJob } from "@/lib/runpod-jobs";
import { buildUploadKeys } from "@/lib/uploads";

const requestSchema = z.object({
  jobId: z.string().uuid(),
  options: z
    .object({
      harmonyThreshold: z.number().min(0).max(1).optional(),
      shadowStrength: z.number().min(0).max(1).optional(),
      reflectionStrength: z.number().min(0).max(1).optional()
    })
    .optional()
});

export async function POST(request: Request): Promise<NextResponse> {
  try {
    const env = getEnv();
    const unauthorized = assertPasscode(request, env.APP_PASSCODE);
    if (unauthorized) {
      return unauthorized;
    }

    const payload = requestSchema.parse(await request.json());
    const setup = await getProvisionedSetupOrThrow();

    const { carKey, backgroundKey } = buildUploadKeys(payload.jobId);
    const outputKey = `outputs/${payload.jobId}/composite.jpg`;

    const [carImageUrl, backgroundImageUrl, outputPutUrl] = await Promise.all([
      presignGet(carKey, 3600),
      presignGet(backgroundKey, 3600),
      presignPut(outputKey, "image/jpeg", 3600)
    ]);

    const runpodJobId = await submitRunpodJob(setup.runpodEndpointId as string, {
      action: "composite",
      job_id: payload.jobId,
      car_image_url: carImageUrl,
      background_image_url: backgroundImageUrl,
      output_put_url: outputPutUrl,
      pipeline_variant: env.PIPELINE_VARIANT,
      options: {
        harmony_threshold: payload.options?.harmonyThreshold ?? 0.65,
        shadow_strength: payload.options?.shadowStrength ?? 0.85,
        reflection_strength: payload.options?.reflectionStrength ?? 0.6
      }
    });

    const now = new Date().toISOString();
    await putJobState({
      jobId: payload.jobId,
      runpodJobId,
      variant: env.PIPELINE_VARIANT,
      input: {
        carKey,
        backgroundKey
      },
      output: {
        outputKey
      },
      createdAt: now,
      updatedAt: now
    });

    return NextResponse.json({
      jobId: payload.jobId,
      status: "queued",
      runpodJobId
    });
  } catch (error) {
    return jsonError(error);
  }
}
