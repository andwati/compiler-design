// Shared data shapes. Every stage output is plain JSON-serialisable data built from these.

export type Span = { start: number; end: number };

export type TokenKind = 'kw' | 'type' | 'ident' | 'int' | 'op' | 'punct' | 'eof';
export interface Token {
  kind: TokenKind;
  text: string;
  start: number;
  end: number;
  line: number;
  col: number;
}
export interface Comment {
  start: number;
  end: number;
  text: string;
  line: number;
  col: number;
}
export interface Lexed {
  source: string;
  tokens: Token[];
  comments: Comment[];
  charMap: number[];
}

export type NodeKind =
  | 'Program'
  | 'Fn'
  | 'Param'
  | 'Block'
  | 'Let'
  | 'Assign'
  | 'While'
  | 'If'
  | 'Return'
  | 'ExprStmt'
  | 'Binary'
  | 'Call'
  | 'Ident'
  | 'Int'
  | 'Bool';
export type Ty = 'i32' | 'bool' | 'unit' | 'fn' | 'error';

/**
 * One AST node type for every kind. The kind-specific fields (name, left, right, stmts ...) live behind
 * the index signature: a full discriminated union would triple the size of the parser for no gain, since
 * the scene only ever walks the tree generically via CHILD_KEYS.
 */
export interface AstNode {
  id: number;
  kind: NodeKind;
  parent: number | null;
  span: Span;
  tokens: [number, number];
  ty?: Ty;
  symbol?: number;
  [field: string]: any;
}

export type TraceEvent =
  | { type: 'enter'; rule: string; nodeId: number | null; tokenIndex: number; minBp?: number }
  | { type: 'exit'; rule: string; nodeId: number | null; tokenIndex: number; becameTail?: boolean }
  | { type: 'consume'; tokenIndex: number }
  | { type: 'pratt'; op: string; bp: number | null; minBp: number; tokenIndex: number; taken: boolean };
export interface Parsed {
  root: AstNode;
  nodeCount: number;
  trace: TraceEvent[];
}

export interface Sym {
  id: number;
  name: string;
  kind: 'fn' | 'local' | 'param';
  ty: Ty;
  mut: boolean;
  nodeId: number;
  scopeId: number;
  depth: number;
  params?: Ty[];
  ret?: Ty;
  extern?: boolean;
}
export type SemaEvent =
  | { type: 'push-scope'; scopeId: number; depth: number; label: string; nodeId: number }
  | { type: 'pop-scope'; scopeId: number; depth: number; symbols: number[] }
  | {
      type: 'define';
      symbol: number;
      name: string;
      kind: string;
      ty: Ty;
      mut: boolean;
      nodeId: number;
      scopeId: number;
      depth: number;
    }
  | { type: 'resolve'; nodeId: number; name: string; symbol: number; fromDepth: number; toDepth: number }
  | { type: 'type'; nodeId: number; ty: Ty };
export interface SemaError {
  message: string;
  code: string;
  nodeId: number;
  start: number;
  end: number;
}
export interface SemaResult {
  ok: boolean;
  errors: SemaError[];
  root: AstNode;
  symbols: Sym[];
  events: SemaEvent[];
}

// ---- IR ----
export type Operand = { k: 'const'; v: number } | { k: 'reg'; name: string };
export type IrOp = 'copy' | 'add' | 'sub' | 'mul' | 'lt' | 'gt' | 'eq' | 'call' | 'ret' | 'br' | 'jmp';
export interface IrInstr {
  id: number;
  op: IrOp;
  node: number;
  text?: string;
  dst?: string | null;
  a?: Operand | null;
  b?: Operand;
  callee?: string;
  args?: Operand[];
  then?: string;
  else?: string;
  target?: string;
}
export interface IrBlock {
  label: string;
  instrs: IrInstr[];
  succ: string[];
  pred: string[];
}
export interface IrFn {
  name: string;
  params: { name: string; ty: Ty; nodeId: number }[];
  ret: Ty;
  blocks: IrBlock[];
  nodeId: number;
}
export interface IrModule {
  functions: IrFn[];
  externs: { name: string; params: Ty[]; ret: Ty }[];
}

export interface RewriteEvent {
  pass: string;
  round: number;
  kind: 'propagate' | 'fold' | 'delete' | 'simplify';
  fn: string;
  block: string;
  instrId: number;
  nodeId: number;
  before: string;
  after: string | null;
  reason: string;
}
