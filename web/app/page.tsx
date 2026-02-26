"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";

type ReadyResponse = {
  ready: boolean;
  phase: "provisioning" | "downloading_models" | "ready" | "error";
  message: string;
};

type StatusSuccess = {
  status: "success";
  outputUrl: string;
  harmonyScore?: number;
  quality?: string | null;
};

type StatusRejected = {
  status: "rejected";
  score: number;
  guidance: string[];
};

type StatusError = {
  status: "error";
  message: string;
};

type StatusProcessing = {
  status: "processing";
};

type JobStatus = StatusSuccess | StatusRejected | StatusError | StatusProcessing;

const PASSCODE_STORAGE_KEY = "carcompose_passcode";

async function parseApiJson(response: Response): Promise<any> {
  const text = await response.text();
  let body: Record<string, unknown> = {};
  if (text.trim().length > 0) {
    try {
      body = JSON.parse(text) as Record<string, unknown>;
    } catch {
      body = { message: text };
    }
  }

  if (!response.ok) {
    const message =
      (typeof body.message === "string" && body.message.trim().length > 0 && body.message) ||
      (typeof body.error === "string" && body.error.trim().length > 0 && body.error) ||
      `Request failed (${response.status})`;
    throw new Error(message);
  }
  return body;
}

export default function HomePage(): JSX.Element {
  const [passcodeInput, setPasscodeInput] = useState("");
  const [passcode, setPasscode] = useState("");

  const [readyState, setReadyState] = useState<"awaiting-passcode" | "checking" | "ready" | "error">(
    "awaiting-passcode"
  );
  const [readyMessage, setReadyMessage] = useState("Enter passcode to initialize system.");

  const [carFile, setCarFile] = useState<File | null>(null);
  const [backgroundFile, setBackgroundFile] = useState<File | null>(null);

  const [jobState, setJobState] = useState<"idle" | "uploading" | "processing" | "success" | "rejected" | "error">(
    "idle"
  );
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [harmonyScore, setHarmonyScore] = useState<number | null>(null);
  const [quality, setQuality] = useState<string | null>(null);
  const [guidance, setGuidance] = useState<string[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const apiFetch = useCallback(
    async (path: string, init?: RequestInit): Promise<any> => {
      const headers = new Headers(init?.headers ?? {});
      headers.set("x-carcompose-passcode", passcode);
      if (!headers.has("Content-Type") && init?.body) {
        headers.set("Content-Type", "application/json");
      }

      try {
        const response = await fetch(path, {
          ...init,
          headers
        });

        return parseApiJson(response);
      } catch (error) {
        if (error instanceof TypeError) {
          throw new Error("Request failed before reaching the API. Check deployment/network and retry.");
        }
        throw error;
      }
    },
    [passcode]
  );

  useEffect(() => {
    const storedPasscode = localStorage.getItem(PASSCODE_STORAGE_KEY);
    if (!storedPasscode) {
      return;
    }

    setPasscode(storedPasscode);
    setPasscodeInput(storedPasscode);
  }, []);

  useEffect(() => {
    if (!passcode) {
      setReadyState("awaiting-passcode");
      setReadyMessage("Enter passcode to initialize system.");
      return;
    }

    let canceled = false;
    let timeoutId: number | null = null;

    const pollReady = async (): Promise<void> => {
      setReadyState("checking");
      try {
        const response = (await apiFetch("/api/ready", { method: "GET" })) as ReadyResponse;
        if (canceled) {
          return;
        }

        setReadyMessage(response.message);
        if (response.ready) {
          setReadyState("ready");
          return;
        }

        if (response.phase === "error") {
          setReadyState("error");
          return;
        }

        timeoutId = window.setTimeout(pollReady, 5000);
      } catch (error) {
        if (canceled) {
          return;
        }

        setReadyState("error");
        setReadyMessage(error instanceof Error ? error.message : "Failed to check system readiness.");
      }
    };

    void pollReady();

    return () => {
      canceled = true;
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [apiFetch, passcode]);

  const canSubmit = useMemo(() => {
    return (
      readyState !== "awaiting-passcode" &&
      readyState !== "error" &&
      carFile &&
      backgroundFile &&
      (jobState === "idle" || jobState === "error" || jobState === "success" || jobState === "rejected")
    );
  }, [backgroundFile, carFile, jobState, readyState]);

  const handlePasscodeSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    if (!passcodeInput.trim()) {
      return;
    }

    const next = passcodeInput.trim();
    localStorage.setItem(PASSCODE_STORAGE_KEY, next);
    setPasscode(next);
  };

  const uploadToPresignedUrl = useCallback(async (label: string, putUrl: string, file: File): Promise<void> => {
    try {
      const response = await fetch(putUrl, {
        method: "PUT",
        headers: { "Content-Type": file.type },
        body: file
      });

      if (!response.ok) {
        throw new Error(`${label} upload failed (${response.status}).`);
      }
    } catch (error) {
      if (error instanceof TypeError) {
        throw new Error(
          `${label} upload failed before reaching storage. If this is the first run, keep the page open until system initialization configures R2 CORS.`
        );
      }
      throw error;
    }
  }, []);

  const handleProcess = async (): Promise<void> => {
    if (!carFile || !backgroundFile) {
      return;
    }

    setErrorMessage(null);
    setResultUrl(null);
    setHarmonyScore(null);
    setQuality(null);
    setGuidance([]);

    try {
      setJobState("uploading");

      const presign = await apiFetch("/api/uploads/presign", {
        method: "POST",
        body: JSON.stringify({
          contentTypes: {
            car: carFile.type,
            background: backgroundFile.type
          }
        })
      });

      await Promise.all([
        uploadToPresignedUrl("Car", presign.car.putUrl as string, carFile),
        uploadToPresignedUrl("Background", presign.background.putUrl as string, backgroundFile)
      ]);

      setJobState("processing");
      const submitted = await apiFetch("/api/composite", {
        method: "POST",
        body: JSON.stringify({
          jobId: presign.jobId,
          options: {
            harmonyThreshold: 0.65,
            shadowStrength: 0.85,
            reflectionStrength: 0.6
          }
        })
      });

      for (let attempt = 0; attempt < 240; attempt += 1) {
        await new Promise((resolve) => setTimeout(resolve, 3000));

        const status = (await apiFetch(`/api/status/${submitted.jobId}`, { method: "GET" })) as JobStatus;
        if (status.status === "processing") {
          continue;
        }

        if (status.status === "error") {
          setJobState("error");
          setErrorMessage(status.message);
          return;
        }

        if (status.status === "rejected") {
          setJobState("rejected");
          setHarmonyScore(status.score);
          setGuidance(status.guidance);
          return;
        }

        if (status.status === "success") {
          setJobState("success");
          setResultUrl(status.outputUrl);
          setHarmonyScore(status.harmonyScore ?? null);
          setQuality(status.quality ?? null);
          return;
        }
      }

      throw new Error("Job timed out after 12 minutes.");
    } catch (error) {
      setJobState("error");
      setErrorMessage(error instanceof Error ? error.message : "Request failed.");
    }
  };

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-5xl flex-col gap-8 px-6 py-10">
      <section className="rounded-2xl border border-black/10 bg-panel/80 p-8 shadow-xl shadow-black/5">
        <p className="mb-2 text-sm uppercase tracking-[0.2em] text-accent">CarCompose MVP</p>
        <h1 className="font-display text-4xl leading-tight text-ink md:text-5xl">
          One-click car composite pipeline
        </h1>
        <p className="mt-4 max-w-3xl text-base text-black/70">
          Upload a car photo and a target background. The system initializes itself and runs the GPU pipeline on
          RunPod.
        </p>
      </section>

      <section className="rounded-2xl border border-black/10 bg-white/70 p-6 shadow-lg shadow-black/5">
        <h2 className="font-display text-2xl text-ink">Access</h2>
        <form onSubmit={handlePasscodeSubmit} className="mt-4 flex flex-col gap-3 md:flex-row">
          <input
            type="password"
            value={passcodeInput}
            onChange={(event) => setPasscodeInput(event.target.value)}
            placeholder="Enter app passcode"
            className="w-full rounded-lg border border-black/20 bg-white px-4 py-3 text-base outline-none ring-accent/40 focus:ring"
          />
          <button
            type="submit"
            className="rounded-lg bg-ink px-5 py-3 text-sm font-semibold uppercase tracking-wide text-white"
          >
            Unlock
          </button>
        </form>
      </section>

      <section className="rounded-2xl border border-black/10 bg-white/70 p-6 shadow-lg shadow-black/5">
        <h2 className="font-display text-2xl text-ink">System State</h2>
        <p className="mt-3 text-sm uppercase tracking-[0.2em] text-black/50">{readyState}</p>
        <p className="mt-2 text-base text-black/75">{readyMessage}</p>
      </section>

      <section className="rounded-2xl border border-black/10 bg-white/70 p-6 shadow-lg shadow-black/5">
        <h2 className="font-display text-2xl text-ink">Upload and Process</h2>
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          <label className="rounded-xl border border-black/15 bg-white p-4">
            <p className="mb-2 text-sm font-semibold uppercase tracking-wide text-black/60">Car Photo</p>
            <input type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => setCarFile(event.target.files?.[0] ?? null)} />
          </label>
          <label className="rounded-xl border border-black/15 bg-white p-4">
            <p className="mb-2 text-sm font-semibold uppercase tracking-wide text-black/60">Background</p>
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp"
              onChange={(event) => setBackgroundFile(event.target.files?.[0] ?? null)}
            />
          </label>
        </div>

        <button
          type="button"
          onClick={() => void handleProcess()}
          disabled={!canSubmit}
          className="mt-6 rounded-lg bg-accent px-5 py-3 text-sm font-semibold uppercase tracking-wide text-white disabled:cursor-not-allowed disabled:opacity-50"
        >
          {jobState === "uploading" && "Uploading"}
          {jobState === "processing" && "Processing"}
          {(jobState === "idle" || jobState === "error" || jobState === "success" || jobState === "rejected") &&
            (readyState === "ready" ? "Process" : "Queue Job")}
        </button>

        {jobState === "processing" && (
          <p className="mt-3 text-sm text-black/70">Processing. Waiting for real pipeline status...</p>
        )}

        {jobState === "success" && resultUrl && (
          <div className="mt-4 rounded-xl border border-green-700/30 bg-green-50 p-4">
            <p className="text-sm font-semibold uppercase tracking-wide text-green-900">Completed</p>
            <p className="mt-2 text-sm text-green-900">Harmony score: {harmonyScore ?? "n/a"}</p>
            <p className="text-sm text-green-900">Quality: {quality ?? "n/a"}</p>
            <a
              href={resultUrl}
              target="_blank"
              rel="noreferrer"
              className="mt-3 inline-block rounded-md bg-green-800 px-4 py-2 text-sm font-semibold text-white"
            >
              Download Composite
            </a>
          </div>
        )}

        {jobState === "rejected" && (
          <div className="mt-4 rounded-xl border border-amber-700/30 bg-amber-50 p-4">
            <p className="text-sm font-semibold uppercase tracking-wide text-amber-900">Rejected</p>
            <p className="mt-2 text-sm text-amber-900">Harmony score: {harmonyScore ?? "n/a"}</p>
            <ul className="mt-2 list-disc pl-5 text-sm text-amber-900">
              {guidance.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        )}

        {jobState === "error" && errorMessage && (
          <div className="mt-4 rounded-xl border border-red-700/30 bg-red-50 p-4">
            <p className="text-sm font-semibold uppercase tracking-wide text-red-900">Error</p>
            <p className="mt-2 text-sm text-red-900">{errorMessage}</p>
          </div>
        )}
      </section>
    </main>
  );
}
