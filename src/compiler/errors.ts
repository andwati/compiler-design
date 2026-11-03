// Shared error type. Carries a source span so the scene can point at the offending characters.
export class CompileError extends Error {
  stage: string;
  detail: string;
  start: number;
  end: number;
  constructor(stage: string, message: string, start = 0, end = start) {
    super(`${stage}: ${message}`);
    this.name = 'CompileError';
    this.stage = stage;
    this.detail = message;
    this.start = start;
    this.end = end;
  }
}
