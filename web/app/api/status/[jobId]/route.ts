import { NextResponse } from "next/server";
import { z } from "zod";
import { assertPasscode } from "@/lib/auth";
import { getEnv } from "@/lib/env";
import { jsonError } from "@/lib/http";
import { presignGet } from "@/lib/r2";
import { getJobState, getSetupState } from "@/lib/r2-state";
import { getRunpodJobStatus } from "@/lib/runpod-jobs";

const paramsSchema = z.object({
  jobId: z.string().uuid()
});

type WorkerSuccessOutput = {
  status: "success";
  harmonyScore?: number;
  timings?: Record<string, number>;
  variant?: "core" | "full";
  quality?: string;
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
    const [setup, job] = await Promise.all([getSetupState(), getJobState(jobId)]);

    if (!setup?.runpodEndpointId) {
      return NextResponse.json(
        { error: "not_ready", message: "System is initializing. Try again soon." },
        { status: 409 }
      );
    }

    if (!job) {
      return NextResponse.json(
        { error: "not_found", message: "Job not found." },
        { status: 404 }
      );
    }

    const runpod = await getRunpodJobStatus(setup.runpodEndpointId, job.runpodJobId);
    if (runpod.status === "IN_QUEUE" || runpod.status === "IN_PROGRESS") {
      return NextResponse.json({ status: "processing" });
    }

    if (runpod.status === "FAILED") {
      return NextResponse.json({ status: "error", message: runpod.error ?? "RunPod job failed." });
    }

    if (runpod.status !== "COMPLETED") {
      return NextResponse.json({ status: "processing" });
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
      quality: success.quality ?? null
    });
  } catch (error) {
    return jsonError(error, 400);
  }
}
