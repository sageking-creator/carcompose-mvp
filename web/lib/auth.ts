import { timingSafeEqual } from "node:crypto";
import { NextResponse } from "next/server";

export const PASSCODE_HEADER = "x-carcompose-passcode";

export function isPasscodeValid(received: string | null, expected: string): boolean {
  if (!received) {
    return false;
  }

  const left = Buffer.from(received);
  const right = Buffer.from(expected);

  if (left.length !== right.length) {
    return false;
  }

  return timingSafeEqual(left, right);
}

export function assertPasscode(request: Request, expected: string): NextResponse | null {
  const received = request.headers.get(PASSCODE_HEADER);
  if (isPasscodeValid(received, expected)) {
    return null;
  }

  return NextResponse.json(
    { error: "unauthorized", message: "Missing or invalid passcode." },
    { status: 401 }
  );
}
