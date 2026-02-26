import { NextResponse } from "next/server";
import { z } from "zod";
import { assertPasscode } from "@/lib/auth";
import { getEnv } from "@/lib/env";
import { jsonError } from "@/lib/http";
import { ensureReady } from "@/lib/ready-service";
import { presignGet, presignPut } from "@/lib/r2";
import { getJobState, getSetupState, patchJobState } from "@/lib/r2-state";
import { getRunpodJobStatus, submitRunpodJob } from "@/lib/runpod-jobs";

const paramsSchema = z.object({
  jobId: z.string().uuid()
});

type WorkerSuccessOutput = {
  status: "success";
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
    maskAreaRatio?: number;
    edgeHaloMeanDelta?: number;
    edgeBandWidthPx?: number;
    protectCoverageRatio?: number;
    contactShadowApplied?: boolean;
    glassModeApplied?: "off" | "auto" | "force";
    studioModeApplied?: "off" | "auto" | "on";
  };
};

type WorkerRejectedOutput = {
  status: "rejected";
  score: number;
  guidance: string[];
};

type WorkerErrorOutput = {
  status: "error";
  message: string;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
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

      const [carImageUrl, backgroundImageUrl, outputPutUrl] = await Promise.all([
        presignGet(job.input.carKey, 3600),
        presignGet(job.input.backgroundKey, 3600),
        presignPut(job.output.outputKey, "image/jpeg", 3600)
      ]);

      const submittedRunpodJobId = await submitRunpodJob(setup.runpodEndpointId, {
        action: "composite",
        job_id: job.jobId,
        car_image_url: carImageUrl,
        background_image_url: backgroundImageUrl,
        output_put_url: outputPutUrl,
        pipeline_variant: job.variant,
        options: {
          harmony_threshold: job.options.harmonyThreshold,
          shadow_strength: job.options.shadowStrength,
          reflection_strength: job.options.reflectionStrength
        }
      });

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
      return NextResponse.json({
        status: "error",
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
      return NextResponse.json({
        status: "rejected",
        score: rejected.score,
        guidance: rejected.guidance
      });
    }

    if ((output as WorkerErrorOutput).status === "error") {
      const workerError = output as WorkerErrorOutput;
      return NextResponse.json({
        status: "error",
        message: workerError.message
      });
    }

    const success = output as WorkerSuccessOutput;
    const outputUrl = await presignGet(job.output.outputKey, 60 * 60 * 24);

    return NextResponse.json({
      status: "success",
      outputUrl,
      harmonyScore: success.harmonyScore,
      timings: success.timings ?? {},
      variant: success.variant ?? job.variant,
      quality: success.quality ?? null,
      detailPreservation: success.detailPreservation ?? null,
      artifactChecks: success.artifactChecks ?? null
    });
  } catch (error) {
    return jsonError(error, 400);
  }
}
