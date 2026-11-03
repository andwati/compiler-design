import { describe, expect, it } from 'vitest';
import { CompileError, MAIN_RS, MATH_RS } from '../src/compiler/index.ts';
import { CHAR_COMMENT, CHAR_SPACE, lex } from '../src/compiler/lexer.ts';
import { parse } from '../src/compiler/parser.ts';
import { analyze } from '../src/compiler/sema.ts';
import { sexpr } from './helpers.ts';

const tokenLines = (src: string) => lex(src).tokens.map((t) => `${t.kind}:${t.text}@${t.line}:${t.col}`);

describe('lexer', () => {
  it('math.rs tokens', () => expect(tokenLines(MATH_RS)).toMatchSnapshot());
  it('main.rs tokens', () => expect(tokenLines(MAIN_RS)).toMatchSnapshot());

  it('maps every character to a token, whitespace or comment', () => {
    const l = lex(MAIN_RS);
    expect(l.charMap).toHaveLength(MAIN_RS.length);
    l.charMap.forEach((m, i) => {
      if (m >= 0) expect(i >= l.tokens[m].start && i < l.tokens[m].end).toBe(true);
      else expect([CHAR_SPACE, CHAR_COMMENT]).toContain(m);
      if (/\s/.test(MAIN_RS[i])) expect([CHAR_SPACE, CHAR_COMMENT]).toContain(m);
    });
    expect(l.comments).toHaveLength(1);
    expect(l.charMap[MAIN_RS.indexOf('// sum')]).toBe(CHAR_COMMENT);
  });

  it('rejects unknown characters with a span', () => {
    expect(() => lex('fn main() { @ }')).toThrow(CompileError);
  });
});

describe('parser', () => {
  it('math.rs AST', () => expect(sexpr(parse(lex(MATH_RS)).root)).toMatchSnapshot());
  it('main.rs AST', () => expect(sexpr(parse(lex(MAIN_RS)).root)).toMatchSnapshot());

  it('Pratt precedence: a + b * c binds the multiplication tighter', () => {
    const p = parse(lex('fn f() -> i32 { 1 + 2 * 3 - 4 }'));
    expect(sexpr(p.root)).toBe('(program\n  (fn f () -> i32 {tail (- (+ 1 (* 2 3)) 4)}))');
  });

  it('comparison binds looser than arithmetic, equality loosest', () => {
    const p = parse(lex('fn f() -> bool { 1 + 2 < 3 * 4 == true }'));
    expect(sexpr(p.root)).toContain('(== (< (+ 1 2) (* 3 4)) true)');
  });

  it('gives every node a unique id, a parent link and a span inside its parent', () => {
    const { root, nodeCount } = parse(lex(MAIN_RS));
    const seen = new Map<number, any>();
    (function visit(n: any, parent: any) {
      expect(seen.has(n.id)).toBe(false);
      seen.set(n.id, n);
      expect(n.parent).toBe(parent ? parent.id : null);
      if (parent) {
        expect(n.span.start).toBeGreaterThanOrEqual(parent.span.start);
        expect(n.span.end).toBeLessThanOrEqual(parent.span.end);
      }
      for (const k of ['items', 'params', 'stmts', 'args']) for (const c of n[k] ?? []) visit(c, n);
      for (const k of [
        'body',
        'tail',
        'init',
        'target',
        'value',
        'cond',
        'then',
        'else',
        'expr',
        'left',
        'right',
        'callee',
      ])
        if (n[k] && typeof n[k] === 'object') visit(n[k], n);
    })(root, null);
    expect(seen.size).toBe(nodeCount);
  });

  it('records a parse trace whose enters and exits balance', () => {
    const { trace } = parse(lex(MAIN_RS));
    let depth = 0;
    for (const e of trace) {
      if (e.type === 'enter') depth++;
      if (e.type === 'exit') depth--;
      expect(depth).toBeGreaterThanOrEqual(0);
    }
    expect(depth).toBe(0);
    expect(trace.filter((e) => e.type === 'consume').length).toBe(lex(MAIN_RS).tokens.length - 1); // every token but EOF
    expect(trace.some((e) => e.type === 'pratt' && e.taken)).toBe(true);
  });

  it('reports syntax errors with a position', () => {
    expect(() => parse(lex('fn main() -> i32 { let x: i32 = ; }'))).toThrow(/expected an expression/);
  });

  it('is JSON-serialisable (no object cycles)', () => {
    const p = parse(lex(MAIN_RS));
    expect(JSON.parse(JSON.stringify(p)).nodeCount).toBe(p.nodeCount);
  });
});

const sema = (src: string) => analyze(parse(lex(src)));

describe('semantic analysis', () => {
  it('accepts both sample files', () => {
    expect(sema(MATH_RS).errors).toEqual([]);
    expect(sema(MAIN_RS).errors).toEqual([]);
  });

  it('rejects assignment to an immutable binding', () => {
    const r = sema('fn main() -> i32 { let x: i32 = 1; x = 2; x }');
    expect(r.ok).toBe(false);
    expect(r.errors[0].message).toMatch(/cannot assign twice to immutable variable `x`/);
    expect(r.errors[0].code).toBe('E0384');
  });

  it('rejects assignment to a parameter', () => {
    expect(sema('fn f(a: i32) -> i32 { a = 1; a }').errors[0].code).toBe('E0384');
  });

  it('rejects type mismatches', () => {
    expect(sema('fn main() -> i32 { let x: i32 = true; x }').errors[0].message).toMatch(/expected i32, found bool/);
    expect(sema('fn main() -> i32 { let x: i32 = 1; if x { } 0 }').errors[0].message).toMatch(
      /expected bool, found i32/,
    );
    expect(sema('fn main() -> bool { 1 + 2 }').errors[0].message).toMatch(/expected bool, found i32/);
    expect(sema('fn main() -> i32 { 1 + true }').errors[0].message).toMatch(/needs i32 operands/);
  });

  it('rejects wrong call arity and argument types', () => {
    expect(sema('fn f(a: i32) -> i32 { a } fn main() -> i32 { f(1, 2) }').errors[0].code).toBe('E0061');
    expect(sema('fn f(a: i32) -> i32 { a } fn main() -> i32 { f(true) }').errors[0].code).toBe('E0308');
  });

  it('rejects unresolved names', () => {
    const r = sema('fn main() -> i32 { y }');
    expect(r.errors[0].message).toMatch(/cannot find value `y`/);
    expect(sema('fn main() -> i32 { nope(1) }').errors[0].code).toBe('E0425');
  });

  it('respects block scoping', () => {
    expect(sema('fn main() -> i32 { if true { let a: i32 = 1; } a }').errors[0].code).toBe('E0425');
  });

  it('lets a let initializer see the outer binding when shadowing', () => {
    expect(sema('fn main() -> i32 { let x: i32 = 1; let x: i32 = x + 1; x }').errors).toEqual([]);
  });

  it('emits the symbol-table story in a consistent order', () => {
    const { events } = sema(MAIN_RS);
    let depth = -1;
    for (const e of events) {
      if (e.type === 'push-scope') {
        expect(e.depth).toBe(depth + 1);
        depth = e.depth;
      }
      if (e.type === 'pop-scope') {
        expect(e.depth).toBe(depth);
        depth = e.depth - 1;
      }
    }
    expect(depth).toBe(-1);
    const resolves = events.filter((e) => e.type === 'resolve');
    expect(resolves.map((e: any) => e.name)).toEqual(expect.arrayContaining(['sum', 'i', 'square', 'k']));
    // `square` is resolved from inside nested scopes back to the global scope (depth 0)
    const sq: any = resolves.find((e: any) => e.name === 'square');
    expect(sq.toDepth).toBe(0);
    expect(sq.fromDepth).toBeGreaterThan(0);
  });

  it('annotates the AST with types and does not mutate the parser output', () => {
    const parsed = parse(lex(MAIN_RS));
    const before = JSON.stringify(parsed);
    const r = analyze(parsed);
    expect(JSON.stringify(parsed)).toBe(before);
    expect(r.root.ty).toBe('unit');
    expect(r.root.items[1].body.tail.ty).toBe('i32');
  });
});
