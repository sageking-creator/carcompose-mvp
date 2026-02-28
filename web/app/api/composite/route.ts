import { NextResponse } from "next/server";
import { z } from "zod";
import { assertPasscode } from "@/lib/auth";
import { getEnv } from "@/lib/env";
import { generateFalBirefnetMask } from "@/lib/fal-birefnet";
import { jsonError } from "@/lib/http";
import { ensureReady } from "@/lib/ready-service";
import { presignGet, presignPut, putBytes } from "@/lib/r2";
import { getSetupState, putJobState } from "@/lib/r2-state";
import { submitRunpodJob } from "@/lib/runpod-jobs";
import {
  buildCompositeRunpodInput,
  buildDebugArtifactKeys,
  buildUploadKeys,
  DEBUG_ARTIFACT_SPECS,
  type DebugArtifactName
} from "@/lib/uploads";

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

type EffectiveMaskBackend = "local" | "fal";

function resolveMaskBackend(env: ReturnType<typeof getEnv>): EffectiveMaskBackend {
  if (env.MASK_BACKEND === "local") {
    return "local";
  }

  if (env.MASK_BACKEND === "fal") {
    if (!env.FAL_KEY || env.FAL_KEY.trim().length === 0) {
      throw new Error("MASK_BACKEND=fal requires FAL_KEY to be set.");
    }
    return "fal";
  }

  return env.FAL_KEY && env.FAL_KEY.trim().length > 0 ? "fal" : "local";
}

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
    let maskBackend = resolveMaskBackend(env);
    let maskKey: string | undefined;
    let cutoutKey: string | undefined;

    if (maskBackend === "fal") {
      const carImageUrlForFal = await presignGet(carKey, 15 * 60);
      try {
        const falResult = await generateFalBirefnetMask({
          apiKey: env.FAL_KEY ?? "",
          imageUrl: carImageUrlForFal,
          model: env.FAL_BIREFNET_MODEL,
          operatingResolution: env.FAL_BIREFNET_OPERATING_RESOLUTION,
          refineForeground: env.FAL_BIREFNET_REFINE_FOREGROUND,
          timeoutSeconds: env.FAL_TIMEOUT_S
        });

        maskKey = `masks/${payload.jobId}/fal_mask.png`;
        await putBytes(maskKey, falResult.maskBytes, falResult.contentType || "image/png");
        if (falResult.cutoutBytes && falResult.cutoutBytes.byteLength > 0) {
          cutoutKey = `masks/${payload.jobId}/fal_cutout.png`;
          await putBytes(cutoutKey, falResult.cutoutBytes, falResult.cutoutContentType || "image/png");
        }
      } catch (error) {
        if (env.MASK_BACKEND === "auto") {
          maskBackend = "local";
          maskKey = undefined;
          cutoutKey = undefined;
        } else {
          throw error;
        }
      }
    }

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
      const [carImageUrl, carMaskUrl, carCutoutUrl, backgroundImageUrl, outputPutUrl, debugPutUrls] = await Promise.all([
        presignGet(carKey, 3600),
        maskKey ? presignGet(maskKey, 3600) : Promise.resolve(undefined),
        cutoutKey ? presignGet(cutoutKey, 3600) : Promise.resolve(undefined),
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

      runpodJobId = await submitRunpodJob(
        endpointId,
        buildCompositeRunpodInput({
          jobId: payload.jobId,
          carImageUrl,
          carMaskUrl,
          carCutoutUrl,
          backgroundImageUrl,
          outputPutUrl,
          pipelineVariant: env.PIPELINE_VARIANT,
          options,
          debugPutUrls
        })
      );
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
        backgroundKey,
        maskKey,
        cutoutKey,
        maskBackend
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
      waitingForInit: !canSubmitNow,
      maskBackend
    });
  } catch (error) {
    return jsonError(error);
  }
}
