import { type Assembled, assemble } from './assembler.js';
import { type FnCode, generate, type Regs } from './codegen.js';
import { type ObjectLayout, writeObject } from './elf.js';
import { CompileError } from './errors.js';
import { lower } from './ir.js';
import { lex } from './lexer.js';
import { type LinkResult, link } from './linker.js';
import { type LoadResult, load, type RunResult, run } from './loader.js';
import { optimize } from './optimizer.js';
import { parse } from './parser.js';
import { MAIN_RS, MATH_RS } from './samples.js';
import { analyze } from './sema.js';
import type { IrModule, Lexed, Parsed, RewriteEvent, SemaResult } from './types.js';
import type { AsmInstr } from './x86.js';
import { imm, reg } from './x86.js';

export { CompileError } from './errors.js';
export * from './samples.js';

export interface UnitResult {
  name: string;
  lexed: Lexed;
  parsed: Parsed;
  sema: SemaResult;
  ir: IrModule;
  optimized: { ir: IrModule; events: RewriteEvent[]; rounds: { fn: string; round: number; events: number }[] };
  code: { functions: FnCode[] };
  assembled: Assembled;
  object: ObjectLayout;
}

export interface CompileOptions {
  regs?: Regs;
}

/** Run one source file through every stage up to and including the ELF object. */
export function compileUnit(name: string, source: string, options: CompileOptions = {}): UnitResult {
  const lexed = lex(source);
  const parsed = parse(lexed);
  const sema = analyze(parsed);
  if (!sema.ok) {
    const e = sema.errors[0];
    throw new CompileError('semantic', e.message, e.start, e.end);
  }
  const ir = lower(sema);
  const optimized = optimize(ir);
  const code = generate(optimized.ir, options);
  const assembled = assemble(code.functions.flatMap((f) => f.asm));
  const defined = new Set(assembled.symbols.map((s) => s.name));
  const undef = [...new Set(assembled.relocs.map((r) => r.symbol).filter((s) => !defined.has(s)))];
  const object = writeObject({
    text: assembled.bytes,
    symbols: assembled.symbols,
    undefined: undef,
    relocs: assembled.relocs.map((r) => ({ offset: r.offset, symbol: r.symbol, addend: r.addend })),
  });
  return { name, lexed, parsed, sema, ir, optimized, code, assembled, object };
}

/** The hand-written C runtime start-up object: call main, exit(main's result). */
export function buildCrt0(): { asm: AsmInstr[]; assembled: Assembled; object: ObjectLayout } {
  const asm: AsmInstr[] = [
    { op: 'label', name: '_start', global: true },
    { op: 'call', sym: 'main' },
    { op: 'mov', size: 32, dst: reg('rdi'), src: reg('rax') }, // exit status = main's return value (edi)
    { op: 'mov', size: 32, dst: reg('rax'), src: imm(60) }, // syscall number 60 = exit
    { op: 'syscall' },
  ];
  const assembled = assemble(asm);
  const object = writeObject({
    text: assembled.bytes,
    symbols: assembled.symbols,
    undefined: ['main'],
    relocs: assembled.relocs.map((r) => ({ offset: r.offset, symbol: r.symbol, addend: r.addend })),
  });
  return { asm, assembled, object };
}

export interface ProgramResult {
  math: UnitResult;
  main: UnitResult;
  crt0: ReturnType<typeof buildCrt0>;
  linked: LinkResult;
  loaded: LoadResult;
  execution: RunResult;
}

/** The full pipeline: source text -> running process. Link order follows the brief: main.o, math.o, crt0.o. */
export function buildProgram(
  sources: { math: string; main: string } = { math: MATH_RS, main: MAIN_RS },
  options: CompileOptions = {},
): ProgramResult {
  const math = compileUnit('math.rs', sources.math, options);
  const main = compileUnit('main.rs', sources.main, options);
  const crt0 = buildCrt0();
  const linked = link([
    { name: 'main.o', bytes: main.object.bytes },
    { name: 'math.o', bytes: math.object.bytes },
    { name: 'crt0.o', bytes: crt0.object.bytes },
  ]);
  const loaded = load(linked.bytes);
  const execution = run(loaded);
  return { math, main, crt0, linked, loaded, execution };
}
