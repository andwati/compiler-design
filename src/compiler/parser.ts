import { CompileError } from './errors.js';
import type { AstNode, Lexed, NodeKind, Parsed, Token, TokenKind, TraceEvent, Ty } from './types.js';

// Which fields of each node kind hold child nodes (used for parent links and walking).
export const CHILD_KEYS: Record<NodeKind, string[]> = {
  Program: ['items'],
  Fn: ['params', 'body'],
  Param: [],
  Block: ['stmts', 'tail'],
  Let: ['init'],
  Assign: ['target', 'value'],
  While: ['cond', 'body'],
  If: ['cond', 'then', 'else'],
  Return: ['value'],
  ExprStmt: ['expr'],
  Binary: ['left', 'right'],
  Call: ['callee', 'args'],
  Ident: [],
  Int: [],
  Bool: [],
};

export function children(node: AstNode): AstNode[] {
  const out: AstNode[] = [];
  for (const k of CHILD_KEYS[node.kind]) {
    const v = node[k];
    if (Array.isArray(v)) out.push(...v);
    else if (v) out.push(v);
  }
  return out;
}

export function walk(node: AstNode, fn: (n: AstNode) => void): void {
  fn(node);
  for (const c of children(node)) walk(c, fn);
}

export function indexNodes(root: AstNode): Map<number, AstNode> {
  const m = new Map<number, AstNode>();
  walk(root, (n) => m.set(n.id, n));
  return m;
}

// Binding powers for the Pratt loop. Higher binds tighter. All left-associative.
const BP: Record<string, number> = { '==': 1, '<': 2, '>': 2, '+': 3, '-': 3, '*': 4 };

/**
 * Recursive-descent parser; expressions use Pratt parsing.
 * Output: AST with node ids, parent ids (no object cycles => serializable) and spans,
 * plus `trace`: the exact order in which productions fired.
 *   enter/exit: a production started/finished (nodeId when it builds a node)
 *   consume:    a token was eaten
 *   pratt:      the Pratt loop compared an operator's binding power with the caller's minimum
 */
export function parse(lexed: Lexed): Parsed {
  const { tokens } = lexed;
  let pos = 0,
    nextId = 0;
  const trace: TraceEvent[] = [];

  const peek = (o = 0): Token => tokens[Math.min(pos + o, tokens.length - 1)];
  const err = (msg: string, t: Token = peek()): never => {
    throw new CompileError('parser', `${msg} (line ${t.line}:${t.col})`, t.start, t.end);
  };
  const is = (kind: TokenKind, text?: string, o = 0) => {
    const t = peek(o);
    return t.kind === kind && (text === undefined || t.text === text);
  };

  function eat(kind: TokenKind, text?: string): Token {
    const t = peek();
    if (t.kind !== kind || (text !== undefined && t.text !== text))
      err(`expected ${text ?? kind}, found '${t.text || 'end of file'}'`);
    trace.push({ type: 'consume', tokenIndex: pos });
    pos++;
    return t;
  }

  function begin(kind: NodeKind, rule: string, startTok: Token = peek()): AstNode {
    const node: AstNode = {
      id: nextId++,
      kind,
      parent: null,
      span: { start: startTok.start, end: startTok.start },
      tokens: [pos, pos],
    };
    trace.push({ type: 'enter', rule, nodeId: node.id, tokenIndex: pos });
    return node;
  }
  function finish(node: AstNode, rule: string): AstNode {
    node.span.end = tokens[pos - 1].end;
    node.tokens[1] = pos - 1;
    trace.push({ type: 'exit', rule, nodeId: node.id, tokenIndex: pos });
    return node;
  }

  function parseType(): Ty {
    const t = peek();
    if (t.kind !== 'type') err(`expected a type, found '${t.text}'`);
    pos++;
    trace.push({ type: 'consume', tokenIndex: pos - 1 });
    return t.text as Ty;
  }

  function parseProgram(): AstNode {
    const n = begin('Program', 'program');
    n.items = [];
    while (!is('eof')) n.items.push(parseFn());
    pos = tokens.length - 1;
    finish(n, 'program');
    n.span = { start: 0, end: lexed.source.length };
    return n;
  }

  function parseFn(): AstNode {
    const n = begin('Fn', 'fn_item');
    n.isExtern = false;
    if (is('kw', 'extern')) {
      eat('kw', 'extern');
      n.isExtern = true;
    }
    eat('kw', 'fn');
    const name = eat('ident');
    n.name = name.text;
    n.nameSpan = { start: name.start, end: name.end };
    eat('punct', '(');
    n.params = [];
    while (!is('punct', ')')) {
      n.params.push(parseParam());
      if (is('punct', ',')) eat('punct', ',');
      else break;
    }
    eat('punct', ')');
    n.ret = 'unit';
    if (is('punct', '->')) {
      eat('punct', '->');
      n.ret = parseType();
    }
    if (n.isExtern) {
      eat('punct', ';');
      n.body = null;
    } else n.body = parseBlock();
    return finish(n, 'fn_item');
  }

  function parseParam(): AstNode {
    const n = begin('Param', 'param');
    const name = eat('ident');
    n.name = name.text;
    n.nameSpan = { start: name.start, end: name.end };
    eat('punct', ':');
    n.ty = parseType();
    return finish(n, 'param');
  }

  function parseBlock(): AstNode {
    const n = begin('Block', 'block');
    n.stmts = [];
    n.tail = null;
    eat('punct', '{');
    while (!is('punct', '}')) {
      if (is('eof')) err('unterminated block');
      const s = parseStmt();
      if (s.tailExpr) {
        n.tail = s.tailExpr;
        break;
      }
      n.stmts.push(s);
    }
    eat('punct', '}');
    return finish(n, 'block');
  }

  // Returns a statement node, or { tailExpr } when an expression is followed by '}' (tail expression).
  function parseStmt(): AstNode | { tailExpr: AstNode } {
    if (is('kw', 'let')) return parseLet();
    if (is('kw', 'while')) return parseWhile();
    if (is('kw', 'if')) return parseIf();
    if (is('kw', 'return')) return parseReturn();
    if (is('ident') && is('op', '=', 1)) return parseAssign();
    // expression statement or tail expression
    // The ExprStmt node is only allocated once we know a ';' follows; before that the production
    // is anonymous (nodeId null) because it may turn out to be the block's tail expression.
    const startTok = peek(),
      startPos = pos;
    trace.push({ type: 'enter', rule: 'expr_stmt', nodeId: null, tokenIndex: pos });
    const expr = parseExpr(0);
    if (is('punct', '}')) {
      trace.push({ type: 'exit', rule: 'expr_stmt', nodeId: null, tokenIndex: pos, becameTail: true });
      return { tailExpr: expr };
    }
    eat('punct', ';');
    const n: AstNode = {
      id: nextId++,
      kind: 'ExprStmt',
      parent: null,
      expr,
      span: { start: startTok.start, end: tokens[pos - 1].end },
      tokens: [startPos, pos - 1],
    };
    trace.push({ type: 'exit', rule: 'expr_stmt', nodeId: n.id, tokenIndex: pos });
    return n;
  }

  function parseLet() {
    const n = begin('Let', 'let_stmt');
    eat('kw', 'let');
    n.mut = false;
    if (is('kw', 'mut')) {
      eat('kw', 'mut');
      n.mut = true;
    }
    const name = eat('ident');
    n.name = name.text;
    n.nameSpan = { start: name.start, end: name.end };
    eat('punct', ':');
    n.ty = parseType();
    eat('op', '=');
    n.init = parseExpr(0);
    eat('punct', ';');
    return finish(n, 'let_stmt');
  }

  function parseAssign() {
    const n = begin('Assign', 'assign_stmt');
    n.target = parseIdent();
    eat('op', '=');
    n.value = parseExpr(0);
    eat('punct', ';');
    return finish(n, 'assign_stmt');
  }

  function parseWhile() {
    const n = begin('While', 'while_stmt');
    eat('kw', 'while');
    n.cond = parseExpr(0);
    n.body = parseBlock();
    return finish(n, 'while_stmt');
  }

  function parseIf() {
    const n = begin('If', 'if_stmt');
    eat('kw', 'if');
    n.cond = parseExpr(0);
    n.then = parseBlock();
    n.else = null;
    if (is('kw', 'else')) {
      eat('kw', 'else');
      n.else = parseBlock();
    }
    return finish(n, 'if_stmt');
  }

  function parseReturn() {
    const n = begin('Return', 'return_stmt');
    eat('kw', 'return');
    n.value = null;
    if (!is('punct', ';')) n.value = parseExpr(0);
    eat('punct', ';');
    return finish(n, 'return_stmt');
  }

  function parseIdent() {
    const n = begin('Ident', 'ident');
    n.name = eat('ident').text;
    return finish(n, 'ident');
  }

  // Pratt loop: parse a primary, then keep absorbing operators whose binding power >= minBp.
  // Left associativity: the right operand is parsed with minBp = bp + 1, so an operator of the
  // same power ends the right operand and is picked up by this loop instead.
  function parseExpr(minBp: number): AstNode {
    trace.push({ type: 'enter', rule: 'expr', nodeId: null, tokenIndex: pos, minBp });
    let left = parsePrimary();
    for (;;) {
      const t = peek();
      const bp = t.kind === 'op' ? BP[t.text] : undefined;
      trace.push({
        type: 'pratt',
        op: t.text,
        bp: bp ?? null,
        minBp,
        tokenIndex: pos,
        taken: bp !== undefined && bp >= minBp,
      });
      if (bp === undefined || bp < minBp) break;
      const n: AstNode = {
        id: nextId++,
        kind: 'Binary',
        parent: null,
        span: { start: left.span.start, end: t.end },
        tokens: [left.tokens[0], pos],
      };
      trace.push({ type: 'enter', rule: 'binary', nodeId: n.id, tokenIndex: pos });
      n.op = t.text;
      eat('op');
      n.left = left;
      n.right = parseExpr(bp + 1);
      n.span.end = tokens[pos - 1].end;
      n.tokens[1] = pos - 1;
      trace.push({ type: 'exit', rule: 'binary', nodeId: n.id, tokenIndex: pos });
      left = n;
    }
    trace.push({ type: 'exit', rule: 'expr', nodeId: left.id, tokenIndex: pos });
    return left;
  }

  function parsePrimary() {
    const t = peek();
    if (t.kind === 'int') {
      const n = begin('Int', 'int_lit');
      eat('int');
      n.value = Number(t.text);
      if (n.value > 2147483647) err('integer literal out of range for i32', t);
      return finish(n, 'int_lit');
    }
    if (t.kind === 'kw' && (t.text === 'true' || t.text === 'false')) {
      const n = begin('Bool', 'bool_lit');
      eat('kw');
      n.value = t.text === 'true';
      return finish(n, 'bool_lit');
    }
    if (t.kind === 'ident') {
      if (is('punct', '(', 1)) return parseCall();
      return parseIdent();
    }
    if (t.kind === 'punct' && t.text === '(') {
      // Parentheses leave no node of their own; they only steer the recursion.
      eat('punct', '(');
      const e = parseExpr(0);
      eat('punct', ')');
      return e;
    }
    return err(`expected an expression, found '${t.text || 'end of file'}'`);
  }

  function parseCall() {
    const n = begin('Call', 'call');
    n.callee = parseIdent();
    eat('punct', '(');
    n.args = [];
    while (!is('punct', ')')) {
      n.args.push(parseExpr(0));
      if (is('punct', ',')) eat('punct', ',');
      else break;
    }
    eat('punct', ')');
    return finish(n, 'call');
  }

  const root = parseProgram();
  // Parent links by id keep the structure cycle-free and JSON-safe.
  (function link(n: AstNode, parentId: number | null) {
    n.parent = parentId;
    for (const c of children(n)) link(c, n.id);
  })(root, null);

  return { root, nodeCount: nextId, trace };
}
