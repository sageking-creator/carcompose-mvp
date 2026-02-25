import { NextResponse } from "next/server";
import { EnvError } from "@/lib/errors";

export function jsonError(error: unknown, fallbackStatus = 500): NextResponse {
  if (error instanceof EnvError) {
    return NextResponse.json(
      { error: "invalid_env", message: error.message },
      { status: 500 }
    );
  }

  if (error instanceof Error) {
    const status = error.message.toLowerCase().includes("initializing") ? 409 : fallbackStatus;
    return NextResponse.json(
      { error: "request_failed", message: error.message },
      { status }
    );
  }

  return NextResponse.json(
    { error: "request_failed", message: "Unknown server error" },
    { status: fallbackStatus }
  );
}
