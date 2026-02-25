import { getJson, putJson } from "@/lib/r2";

export const SETUP_STATE_KEY = "system/setup.json";

export type InitJobStatus = "NOT_STARTED" | "RUNNING" | "COMPLETED" | "FAILED";

export type SetupState = {
  bucketName: string;
  workerImage: string;
  provisioningHash?: string;
  runpodVolumeId?: string;
  runpodVolumeDatacenterId?: string;
  runpodTemplateId?: string;
  runpodEndpointId?: string;
  initJobId?: string;
  initJobStatus: InitJobStatus;
  updatedAt: string;
};

export type JobState = {
  jobId: string;
  runpodJobId: string;
  variant: "core" | "full";
  input: {
    carKey: string;
    backgroundKey: string;
  };
  output: {
    outputKey: string;
  };
  createdAt: string;
  updatedAt: string;
};

export async function getSetupState(): Promise<SetupState | null> {
  return getJson<SetupState>(SETUP_STATE_KEY);
}

export async function putSetupState(patch: Partial<SetupState>): Promise<SetupState> {
  const now = new Date().toISOString();
  const current = (await getSetupState()) ?? {
    bucketName: "",
    workerImage: "",
    initJobStatus: "NOT_STARTED",
    updatedAt: now
  };

  const next: SetupState = {
    ...current,
    ...patch,
    updatedAt: now
  };

  await putJson(SETUP_STATE_KEY, next);
  return next;
}

export async function getJobState(jobId: string): Promise<JobState | null> {
  return getJson<JobState>(`jobs/${jobId}.json`);
}

export async function putJobState(state: JobState): Promise<void> {
  await putJson(`jobs/${state.jobId}.json`, state);
}

export async function patchJobState(jobId: string, patch: Partial<JobState>): Promise<JobState | null> {
  const current = await getJobState(jobId);
  if (!current) {
    return null;
  }

  const next: JobState = {
    ...current,
    ...patch,
    updatedAt: new Date().toISOString()
  };
  await putJobState(next);
  return next;
}
