import { NextResponse } from "next/server";
import { z } from "zod";
import { assertPasscode } from "@/lib/auth";
import { getEnv } from "@/lib/env";
import { jsonError } from "@/lib/http";
import { ensureReady } from "@/lib/ready-service";
import { presignGet, presignPut } from "@/lib/r2";
import { getJobState, getSetupState, patchJobState } from "@/lib/r2-state";
import { getRunpodJobStatus, submitRunpodJob } from "@/lib/runpod-jobs";
import {
  buildCompositeRunpodInput,
  debugArtifactEntries,
  DEBUG_ARTIFACT_SPECS,
  type DebugArtifactName
} from "@/lib/uploads";

const paramsSchema = z.object({
  jobId: z.string().uuid()
});

type WorkerSuccessOutput = {
  status: "success";
  workerBuildId?: string;
  harmonyScore?: number;
  timings?: Record<string, number>;
  variant?: "core" | "full";
  quality?: string;
  detailPreservation?: {
    hfRatio?: number;
    method?: string;
    fallbackReason?: string;
  };
  artifactChecks?: {
    interiorOpaqueRatio?: number;
    outsideLeakMeanAlpha?: number;
    nearLeakMeanAlpha?: number;
    nearLeakP95Alpha?: number;
    maskAreaRatio?: number;
    rawFringeRgbMean?: number;
    rawFringeRgbP95?: number;
    fringeRgbMean?: number;
    fringeRgbP95?: number;
    edgeHaloMeanDelta?: number;
    edgeBandWidthPx?: number;
    protectCoverageRatio?: number;
    contactShadowApplied?: boolean;
    glassModeApplied?: "off" | "auto" | "force";
    glassBackendApplied?: "none" | "legacy" | "sam2";
    studioModeApplied?: "off" | "auto" | "on";
  };
};

type DebugUrls = Partial<Record<DebugArtifactName, string>>;

type WorkerRejectedOutput = {
  status: "rejected";
  workerBuildId?: string;
  score: number;
  guidance: string[];
};

type WorkerErrorOutput = {
  status: "error";
  workerBuildId?: string;
  message: string;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

async function presignDebugUrls(
  debugKeys: Partial<Record<DebugArtifactName, string>> | undefined
): Promise<DebugUrls | null> {
  const entries = debugArtifactEntries(debugKeys);
  if (entries.length === 0) {
    return null;
  }

  const signed = await Promise.all(
    entries.map(
      async ([artifactName, key]): Promise<[DebugArtifactName, string]> => [
        artifactName,
        await presignGet(key, 60 * 60 * 24)
      ]
    )
  );
  return Object.fromEntries(signed) as DebugUrls;
}

export async function GET(
  request: Request,
  context: { params: { jobId: string } }
): Promise<NextResponse> {
  try {
    const env = getEnv();
    const unauthorized = assertPasscode(request, env.APP_PASSCODE);
    if (unauthorized) {
      return unauthorized;
    }

    const { jobId } = paramsSchema.parse(context.params);
    let [setup, job] = await Promise.all([getSetupState(), getJobState(jobId)]);

    if (!job) {
      return NextResponse.json(
        { error: "not_found", message: "Job not found." },
        { status: 404 }
      );
    }

    if (!job.runpodJobId) {
      await ensureReady(env);
      [setup, job] = await Promise.all([getSetupState(), getJobState(jobId)]);
      if (!job) {
        return NextResponse.json(
          { error: "not_found", message: "Job not found." },
          { status: 404 }
        );
      }

      if (!setup?.runpodEndpointId || setup.initJobStatus !== "COMPLETED") {
        return NextResponse.json({ status: "processing" });
      }

      const [carImageUrl, carMaskUrl, backgroundImageUrl, outputPutUrl, debugPutUrls] = await Promise.all([
        presignGet(job.input.carKey, 3600),
        job.input.maskKey ? presignGet(job.input.maskKey, 3600) : Promise.resolve(undefined),
        presignGet(job.input.backgroundKey, 3600),
        presignPut(job.output.outputKey, "image/jpeg", 3600),
        debugArtifactEntries(job.output.debugKeys).length > 0
          ? Promise.all(
              debugArtifactEntries(job.output.debugKeys).map(
                async ([artifactName, key]): Promise<[DebugArtifactName, string]> => [
                  artifactName,
                  await presignPut(key, DEBUG_ARTIFACT_SPECS[artifactName].contentType, 3600)
                ]
              )
            ).then((entries) => Object.fromEntries(entries) as DebugUrls)
          : Promise.resolve(undefined)
      ]);

      const submittedRunpodJobId = await submitRunpodJob(
        setup.runpodEndpointId,
        buildCompositeRunpodInput({
          jobId: job.jobId,
          carImageUrl,
          carMaskUrl,
          backgroundImageUrl,
          outputPutUrl,
          pipelineVariant: job.variant,
          options: job.options,
          debugPutUrls
        })
      );

      const patched = await patchJobState(jobId, {
        runpodJobId: submittedRunpodJobId,
        runpodEndpointId: setup.runpodEndpointId
      });

      job = patched ?? job;
      if (!job.runpodJobId) {
        return NextResponse.json({ status: "processing" });
      }
    }

    const endpointId = job.runpodEndpointId ?? setup?.runpodEndpointId;
    if (!endpointId) {
      return NextResponse.json({ status: "processing" });
    }

    const runpod = await getRunpodJobStatus(endpointId, job.runpodJobId);
    if (runpod.status === "IN_QUEUE" || runpod.status === "IN_PROGRESS") {
      return NextResponse.json({ status: "processing" });
    }

    if (runpod.status === "RETRYING") {
      return NextResponse.json({ status: "processing" });
    }

    if (
      runpod.status === "FAILED" ||
      runpod.status === "CANCELLED" ||
      runpod.status === "TIMED_OUT" ||
      runpod.status === "ABORTED"
    ) {
      const reason = runpod.status.toLowerCase();
      const debugUrls = await presignDebugUrls(job.output.debugKeys);
      return NextResponse.json({
        status: "error",
        expectedWorkerImage: setup?.workerImage ?? null,
        expectedWorkerImageDigest: setup?.workerImageDigest ?? null,
        debugUrls,
        message: runpod.error ?? `RunPod job ${reason}.`
      });
    }

    if (runpod.status !== "COMPLETED") {
      return NextResponse.json({
        status: "error",
        message: `Unexpected RunPod job status: ${runpod.status}`
      });
    }

    const output = isRecord(runpod.output) ? runpod.output : {};

    if ((output as WorkerRejectedOutput).status === "rejected") {
      const rejected = output as WorkerRejectedOutput;
      const debugUrls = await presignDebugUrls(job.output.debugKeys);
      return NextResponse.json({
        status: "rejected",
        workerBuildId: rejected.workerBuildId ?? null,
        expectedWorkerImage: setup?.workerImage ?? null,
        expectedWorkerImageDigest: setup?.workerImageDigest ?? null,
        score: rejected.score,
        guidance: rejected.guidance,
        debugUrls
      });
    }

    if ((output as WorkerErrorOutput).status === "error") {
      const workerError = output as WorkerErrorOutput;
      const debugUrls = await presignDebugUrls(job.output.debugKeys);
      return NextResponse.json({
        status: "error",
        workerBuildId: workerError.workerBuildId ?? null,
        expectedWorkerImage: setup?.workerImage ?? null,
        expectedWorkerImageDigest: setup?.workerImageDigest ?? null,
        debugUrls,
        message: workerError.message
      });
    }

    const success = output as WorkerSuccessOutput;
    const [outputUrl, debugUrls] = await Promise.all([
      presignGet(job.output.outputKey, 60 * 60 * 24),
      presignDebugUrls(job.output.debugKeys)
    ]);

    return NextResponse.json({
      status: "success",
      jobId: job.jobId,
      outputUrl,
      workerBuildId: success.workerBuildId ?? null,
      expectedWorkerImage: setup?.workerImage ?? null,
      expectedWorkerImageDigest: setup?.workerImageDigest ?? null,
      harmonyScore: success.harmonyScore,
      timings: success.timings ?? {},
      variant: success.variant ?? job.variant,
      quality: success.quality ?? null,
      detailPreservation: success.detailPreservation ?? null,
      artifactChecks: success.artifactChecks ?? null,
      debugUrls
    });
  } catch (error) {
    return jsonError(error, 400);
  }
}
