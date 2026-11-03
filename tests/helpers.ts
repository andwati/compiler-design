import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { Regs } from '../src/compiler/codegen.ts';
import { buildCrt0, compileUnit } from '../src/compiler/index.ts';
import { link } from '../src/compiler/linker.ts';
import { load, run } from '../src/compiler/loader.ts';
import type { AstNode } from '../src/compiler/types.ts';

/** Compile a single-file program (must define main), link with crt0, execute, return the exit code. */
export function exitCodeOf(src: string, regs?: Regs): number {
  const u = compileUnit('t.rs', src, { regs });
  const exe = link([
    { name: 't.o', bytes: u.object.bytes },
    { name: 'crt0.o', bytes: buildCrt0().object.bytes },
  ]);
  return run(load(exe.bytes.slice())).exitCode;
}

/** Compact s-expression rendering of an AST, for readable snapshots. */
export function sexpr(n: AstNode): string {
  switch (n.kind) {
    case 'Program':
      return `(program\n${n.items.map((i: AstNode) => `  ${sexpr(i)}`).join('\n')})`;
    case 'Fn':
      return `(${n.isExtern ? 'extern-fn' : 'fn'} ${n.name} (${n.params.map(sexpr).join(' ')}) -> ${n.ret}${n.body ? ` ${sexpr(n.body)}` : ''})`;
    case 'Param':
      return `${n.name}:${n.ty}`;
    case 'Block':
      return `{${[...n.stmts.map(sexpr), ...(n.tail ? [`tail ${sexpr(n.tail)}`] : [])].join(' ')}}`;
    case 'Let':
      return `(let${n.mut ? ' mut' : ''} ${n.name}:${n.ty} ${sexpr(n.init)})`;
    case 'Assign':
      return `(= ${n.target.name} ${sexpr(n.value)})`;
    case 'While':
      return `(while ${sexpr(n.cond)} ${sexpr(n.body)})`;
    case 'If':
      return `(if ${sexpr(n.cond)} ${sexpr(n.then)}${n.else ? ` ${sexpr(n.else)}` : ''})`;
    case 'Return':
      return `(return${n.value ? ` ${sexpr(n.value)}` : ''})`;
    case 'ExprStmt':
      return `(expr ${sexpr(n.expr)})`;
    case 'Binary':
      return `(${n.op} ${sexpr(n.left)} ${sexpr(n.right)})`;
    case 'Call':
      return `(call ${n.callee.name}${n.args.map((a: AstNode) => ` ${sexpr(a)}`).join('')})`;
    case 'Ident':
      return n.name;
    case 'Int':
      return String(n.value);
    case 'Bool':
      return String(n.value);
  }
}

export const hasTool = (t: string) => spawnSync('which', [t]).status === 0;

/** Write bytes to a temp file and return its path (for objdump / readelf cross-checks). */
export function tmpFile(name: string, bytes: number[]): string {
  const dir = mkdtempSync(join(tmpdir(), 'cc-'));
  const p = join(dir, name);
  writeFileSync(p, Uint8Array.from(bytes));
  return p;
}
export const run2 = (cmd: string, args: string[]) => execFileSync(cmd, args, { encoding: 'utf8' });
