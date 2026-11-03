import { describe, expect, it } from 'vitest';
import { MAIN_RS } from '../src/compiler/index.ts';
import { lower } from '../src/compiler/ir.ts';
import { lex } from '../src/compiler/lexer.ts';
import { optimize } from '../src/compiler/optimizer.ts';
import { parse } from '../src/compiler/parser.ts';
import { analyze } from '../src/compiler/sema.ts';

const irOf = (src: string) => lower(analyze(parse(lex(src))));
const dump = (m: ReturnType<typeof irOf>, fn = 0) => m.functions[fn].blocks.flatMap((b) => b.instrs.map((i) => i.text));

describe('IR', () => {
  it('lowers main.rs into the expected CFG', () => {
    const ir = irOf(MAIN_RS).functions[0];
    expect(ir.blocks.map((b) => b.label)).toEqual(['entry', 'while.cond', 'while.body', 'while.exit']);
    const by = Object.fromEntries(ir.blocks.map((b) => [b.label, b]));
    expect(by['while.cond'].pred.sort()).toEqual(['entry', 'while.body']);
    expect(by['while.cond'].succ).toEqual(['while.body', 'while.exit']);
    expect(dump(irOf(MAIN_RS))).toMatchInlineSnapshot(`
      [
        "sum = copy 0",
        "i = copy 1",
        "unused = mul 7, 6",
        "jmp while.cond",
        "t0 = lt i, 5",
        "br t0, while.body, while.exit",
        "t1 = call square(i)",
        "sum = add sum, t1",
        "i = add i, 1",
        "jmp while.cond",
        "k = add 2, 3",
        "t2 = add sum, k",
        "ret t2",
      ]
    `);
  });

  it('prunes code after return', () => {
    const m = irOf('fn f() -> i32 { return 1; let x: i32 = 2; x }');
    expect(dump(m)).toEqual(['ret 1']);
  });
});

describe('optimizer', () => {
  const opt = optimize(irOf(MAIN_RS));
  const after = dump(opt.ir);

  it('folds 2 + 3 to 5 and 7 * 6 to 42', () => {
    const folds = opt.events.filter((e) => e.kind === 'fold');
    expect(folds.map((e) => [e.before, e.after])).toEqual([
      ['unused = mul 7, 6', 'unused = copy 42'],
      ['k = add 2, 3', 'k = copy 5'],
    ]);
    expect(folds[0].reason).toBe('7 * 6 = 42 at compile time');
  });

  it('propagates k = 5 into the final add', () => {
    const p = opt.events.find((e) => e.kind === 'propagate')!;
    expect(p.before).toBe('t2 = add sum, k');
    expect(p.after).toBe('t2 = add sum, 5');
  });

  it('removes the unused binding (and k, once propagated)', () => {
    const del = opt.events.filter((e) => e.kind === 'delete').map((e) => e.before);
    expect(del).toContain('unused = copy 42');
    expect(del).toContain('k = copy 5');
    expect(after.some((t) => t?.includes('unused'))).toBe(false);
    expect(after.some((t) => t?.includes('mul 7'))).toBe(false);
  });

  it('does NOT propagate loop-carried variables (i, sum are not constants inside the loop)', () => {
    expect(after).toContain('t0 = lt i, 5');
    expect(after).toContain('sum = add sum, t1');
    expect(after).toContain('i = add i, 1');
  });

  it('final IR', () => {
    expect(after).toEqual([
      'sum = copy 0',
      'i = copy 1',
      'jmp while.cond',
      't0 = lt i, 5',
      'br t0, while.body, while.exit',
      't1 = call square(i)',
      'sum = add sum, t1',
      'i = add i, 1',
      'jmp while.cond',
      't2 = add sum, 5',
      'ret t2',
    ]);
  });

  it('every event is a { before, after, reason } record with a pass name', () => {
    for (const e of opt.events) {
      expect(typeof e.before).toBe('string');
      expect(e.reason.length).toBeGreaterThan(0);
      expect(['constant-propagation', 'constant-folding', 'dead-code-elimination']).toContain(e.pass);
    }
  });

  it('folds a constant branch and drops the dead arm', () => {
    const o = optimize(
      irOf('fn f() -> i32 { let c: bool = true; let mut r: i32 = 0; if c { r = 1; } else { r = 2; } r }'),
    );
    expect(o.ir.functions[0].blocks.some((b) => b.label === 'if.else')).toBe(false);
    expect(o.events.some((e) => e.reason === 'branch condition is a constant')).toBe(true);
  });

  it('keeps a call whose result is unused (side effects)', () => {
    const o = optimize(irOf('fn g() -> i32 { 1 } fn f() -> i32 { let x: i32 = g(); 0 }'));
    expect(dump(o.ir, 1)).toEqual(['call g()', 'ret 0']);
  });

  it('does not mutate its input', () => {
    const ir = irOf(MAIN_RS);
    const snap = JSON.stringify(ir);
    optimize(ir);
    expect(JSON.stringify(ir)).toBe(snap);
  });
});
