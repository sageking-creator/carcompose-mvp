import { NextResponse } from "next/server";
import { assertPasscode } from "@/lib/auth";
import { getEnv } from "@/lib/env";
import { jsonError } from "@/lib/http";
import { ensureReady } from "@/lib/ready-service";

export async function GET(request: Request): Promise<NextResponse> {
  try {
    const env = getEnv();
    const unauthorized = assertPasscode(request, env.APP_PASSCODE);
    if (unauthorized) {
      return unauthorized;
    }

    const result = await ensureReady(env);
    return NextResponse.json(result);
  } catch (error) {
    return jsonError(error);
  }
}
