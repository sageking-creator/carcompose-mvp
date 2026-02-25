export class EnvError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "EnvError";
  }
}

export class ApiError extends Error {
  public readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}
