import { NextResponse } from "next/server";
import { z } from "zod";
import { assertPasscode } from "@/lib/auth";
import { getEnv } from "@/lib/env";
import { jsonError } from "@/lib/http";
import { ensureReady } from "@/lib/ready-service";
import { presignGet, presignPut } from "@/lib/r2";
import { getSetupState, putJobState } from "@/lib/r2-state";
import { submitRunpodJob } from "@/lib/runpod-jobs";
import { buildDebugArtifactKeys, buildUploadKeys, DEBUG_ARTIFACT_SPECS, type DebugArtifactName } from "@/lib/uploads";

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
    await ensureReady(env);
    const setup = await getSetupState();

    const { carKey, backgroundKey } = buildUploadKeys(payload.jobId);
    const outputKey = `outputs/${payload.jobId}/composite.jpg`;
    const debugKeys = env.DEBUG_ARTIFACTS ? buildDebugArtifactKeys(payload.jobId) : undefined;
    const options = {
      harmonyThreshold: payload.options?.harmonyThreshold ?? 0.65,
      shadowStrength: payload.options?.shadowStrength ?? 0.85,
      reflectionStrength: payload.options?.reflectionStrength ?? 0.6
    };

    const endpointId = setup?.runpodEndpointId;
    const canSubmitNow = Boolean(endpointId && setup?.initJobStatus === "COMPLETED");
    let runpodJobId: string | undefined;
    let runpodEndpointId: string | undefined;

    if (canSubmitNow && endpointId) {
      const [carImageUrl, backgroundImageUrl, outputPutUrl, debugPutUrls] = await Promise.all([
        presignGet(carKey, 3600),
        presignGet(backgroundKey, 3600),
        presignPut(outputKey, "image/jpeg", 3600),
        debugKeys
          ? Promise.all(
              (Object.entries(debugKeys) as Array<[DebugArtifactName, string]>).map(
                async ([artifactName, key]): Promise<[DebugArtifactName, string]> => [
                  artifactName,
                  await presignPut(key, DEBUG_ARTIFACT_SPECS[artifactName].contentType, 3600)
                ]
              )
            ).then((entries) => Object.fromEntries(entries) as Record<DebugArtifactName, string>)
          : Promise.resolve(undefined)
      ]);

      runpodJobId = await submitRunpodJob(endpointId, {
        action: "composite",
        job_id: payload.jobId,
        car_image_url: carImageUrl,
        background_image_url: backgroundImageUrl,
        output_put_url: outputPutUrl,
        pipeline_variant: env.PIPELINE_VARIANT,
        options: {
          harmony_threshold: options.harmonyThreshold,
          shadow_strength: options.shadowStrength,
          reflection_strength: options.reflectionStrength
        },
        debug_put_urls: debugPutUrls
      });
      runpodEndpointId = endpointId;
    }

    const now = new Date().toISOString();
    await putJobState({
      jobId: payload.jobId,
      runpodJobId,
      runpodEndpointId,
      variant: env.PIPELINE_VARIANT,
      input: {
        carKey,
        backgroundKey
      },
      output: {
        outputKey,
        debugKeys
      },
      options,
      createdAt: now,
      updatedAt: now
    });

    return NextResponse.json({
      jobId: payload.jobId,
      status: "queued",
      runpodJobId,
      waitingForInit: !canSubmitNow
    });
  } catch (error) {
    return jsonError(error);
  }
}
