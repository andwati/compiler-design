/**
 * IR: three-address code in basic blocks, non-SSA.
 *
 * Why non-SSA: every `let`/parameter is one named virtual register that may be assigned many
 * times (the loop counter really is reassigned), and temporaries `tN` are assigned once.
 * Constant propagation then needs a proper dataflow fixpoint over the CFG - the optimizer
 * does that - so folding `i < 5` into `1 < 5` is correctly *refused* inside the loop.
 *
 * Operands:  { k:'const', v } | { k:'reg', name }
 * Instrs:    copy | add | sub | mul | lt | gt | eq | call | ret | br | jmp
 */
import type { AstNode, IrBlock, IrFn, IrInstr, IrModule, IrOp, Operand, SemaResult, Ty } from './types.js';

export const C = (v: number): Operand => ({ k: 'const', v });
export const R = (name: string): Operand => ({ k: 'reg', name });

export const BINOPS: Record<string, IrOp> = { '+': 'add', '-': 'sub', '*': 'mul', '<': 'lt', '>': 'gt', '==': 'eq' };
export const OP_SYMBOL: Record<string, string> = { add: '+', sub: '-', mul: '*', lt: '<', gt: '>', eq: '==' };
export const isBinop = (op: string) => op in OP_SYMBOL;
export const isCompare = (op: string) => op === 'lt' || op === 'gt' || op === 'eq';

const opnd = (o: Operand) => (o.k === 'const' ? String(o.v) : o.name);

export function irText(ins: IrInstr): string {
  switch (ins.op) {
    case 'copy':
      return `${ins.dst} = copy ${opnd(ins.a!)}`;
    case 'call':
      return `${ins.dst ? `${ins.dst} = ` : ''}call ${ins.callee}(${ins.args!.map(opnd).join(', ')})`;
    case 'ret':
      return ins.a ? `ret ${opnd(ins.a)}` : 'ret';
    case 'br':
      return `br ${opnd(ins.a!)}, ${ins.then}, ${ins.else}`;
    case 'jmp':
      return `jmp ${ins.target}`;
    default:
      return `${ins.dst} = ${ins.op} ${opnd(ins.a!)}, ${opnd(ins.b!)}`;
  }
}

// Registers read / written by an instruction (used by liveness in optimizer and codegen).
export function uses(ins: IrInstr): string[] {
  const ops = ins.op === 'call' ? ins.args! : [ins.a, ins.b];
  return ops.flatMap((o) => (o && o.k === 'reg' ? [o.name] : []));
}
export const defOf = (ins: IrInstr) => ins.dst ?? null;
export const isTerminator = (ins: IrInstr) => ins.op === 'ret' || ins.op === 'br' || ins.op === 'jmp';

export function successors(block: IrBlock): string[] {
  const t = block.instrs.at(-1);
  if (!t) return [];
  if (t.op === 'jmp') return [t.target!];
  if (t.op === 'br') return t.then === t.else ? [t.then!] : [t.then!, t.else!];
  return [];
}

/**
 * Recompute succ/pred lists, drop unreachable blocks and put blocks in reverse postorder.
 * RPO with the *true* edge visited last gives the natural layout (cond, body..., exit),
 * so the code generator can fall through instead of emitting extra jumps.
 */
export function rebuildCfg(fn: IrFn): IrFn {
  const byLabel = new Map(fn.blocks.map((b) => [b.label, b]));
  const seen = new Set<string>();
  const post: IrBlock[] = [];
  (function dfs(l: string) {
    if (seen.has(l)) return;
    seen.add(l);
    for (const s of [...successors(byLabel.get(l)!)].reverse()) dfs(s);
    post.push(byLabel.get(l)!);
  })(fn.blocks[0].label);
  fn.blocks = post.reverse();
  for (const b of fn.blocks) {
    b.succ = successors(b);
    b.pred = [];
  }
  for (const b of fn.blocks) for (const s of b.succ) byLabel.get(s)!.pred.push(b.label);
  for (const b of fn.blocks) for (const ins of b.instrs) ins.text = irText(ins);
  return fn;
}

/** Lower the annotated AST to IR. */
export function lower(sema: SemaResult): IrModule {
  const fns: IrFn[] = [];
  const externs: IrModule['externs'] = [];
  for (const f of sema.root.items) {
    if (f.isExtern) {
      externs.push({ name: f.name, params: f.params.map((p: AstNode) => p.ty as Ty), ret: f.ret });
      continue;
    }
    fns.push(lowerFn(f, sema));
  }
  return { functions: fns, externs };
}

function lowerFn(f: AstNode, _sema: SemaResult): IrFn {
  const fn: IrFn = { name: f.name, params: [], ret: f.ret, blocks: [], nodeId: f.id };
  const regOfSymbol = new Map<number, string>(); // symbol id -> unique vreg name
  const usedNames = new Set<string>();
  let nTemp = 0,
    nIns = 0;
  const labelCount: Record<string, number> = {};

  const uniqueName = (base: string) => {
    let n = base,
      k = 1;
    while (usedNames.has(n)) n = `${base}.${k++}`;
    usedNames.add(n);
    return n;
  };
  const newTemp = () => {
    const n = `t${nTemp++}`;
    usedNames.add(n);
    return n;
  };
  const newLabel = (base: string) => {
    const k = labelCount[base] ?? 0;
    labelCount[base] = k + 1;
    return k ? `${base}.${k}` : base;
  };

  let cur!: IrBlock;
  const newBlock = (base: string) => {
    const b: IrBlock = { label: newLabel(base), instrs: [], succ: [], pred: [] };
    fn.blocks.push(b);
    return b;
  };
  const terminated = () => cur.instrs.length > 0 && isTerminator(cur.instrs.at(-1)!);
  const emit = (ins: Partial<IrInstr> & { op: IrOp }, node: AstNode): IrInstr => {
    if (terminated()) cur = newBlock('dead'); // code after `return`: lands in an unreachable block that gets pruned
    const full = { ...ins, id: nIns++, node: node.id } as IrInstr;
    cur.instrs.push(full);
    return full;
  };

  for (const p of f.params) {
    const name = uniqueName(p.name);
    regOfSymbol.set(p.symbol!, name);
    fn.params.push({ name, ty: p.ty, nodeId: p.id });
  }

  // `dst` is a destination hint: `sum = sum + x` writes `sum` directly instead of via a temp + copy.
  function expr(n: AstNode, dst: string | null = null): Operand | null {
    switch (n.kind) {
      case 'Int':
        return C(n.value);
      case 'Bool':
        return C(n.value ? 1 : 0);
      case 'Ident':
        return R(regOfSymbol.get(n.symbol!)!);
      case 'Binary': {
        const a = expr(n.left)!,
          b = expr(n.right)!;
        const d = dst ?? newTemp();
        emit({ op: BINOPS[n.op], dst: d, a, b }, n);
        return R(d);
      }
      case 'Call': {
        const args = n.args.map((x: AstNode) => expr(x)!);
        const returnsValue = n.ty !== 'unit';
        const d = returnsValue ? (dst ?? newTemp()) : null;
        emit({ op: 'call', dst: d, callee: n.callee.name, args }, n);
        return d ? R(d) : null;
      }
      default:
        throw new Error(`lower: unexpected expr ${n.kind}`);
    }
  }

  function assignTo(name: string, valueNode: AstNode, node: AstNode) {
    const v = expr(valueNode, name)!;
    if (!(v.k === 'reg' && v.name === name)) emit({ op: 'copy', dst: name, a: v }, node);
  }

  function blockBody(b: AstNode) {
    for (const s of b.stmts) stmt(s);
    return b.tail ? expr(b.tail) : null;
  }

  function stmt(s: AstNode): void {
    switch (s.kind) {
      case 'Let': {
        const name = uniqueName(s.name);
        regOfSymbol.set(s.symbol!, name);
        assignTo(name, s.init, s);
        break;
      }
      case 'Assign':
        assignTo(regOfSymbol.get(s.target.symbol)!, s.value, s);
        break;
      case 'ExprStmt':
        expr(s.expr);
        break;
      case 'Return':
        emit({ op: 'ret', a: s.value ? expr(s.value) : null }, s);
        break;
      case 'While': {
        const cond = newBlock('while.cond'),
          body = newBlock('while.body'),
          exit = newBlock('while.exit');
        emit({ op: 'jmp', target: cond.label }, s);
        cur = cond;
        const c = expr(s.cond);
        emit({ op: 'br', a: c!, then: body.label, else: exit.label }, s.cond);
        cur = body;
        blockBody(s.body);
        if (!terminated()) emit({ op: 'jmp', target: cond.label }, s);
        cur = exit;
        break;
      }
      case 'If': {
        const th = newBlock('if.then');
        const el = s.else ? newBlock('if.else') : null;
        const join = newBlock('if.end');
        const c = expr(s.cond);
        emit({ op: 'br', a: c!, then: th.label, else: (el ?? join).label }, s.cond);
        cur = th;
        blockBody(s.then);
        if (!terminated()) emit({ op: 'jmp', target: join.label }, s);
        if (el) {
          cur = el;
          blockBody(s.else);
          if (!terminated()) emit({ op: 'jmp', target: join.label }, s);
        }
        cur = join;
        break;
      }
      default:
        throw new Error(`lower: unexpected stmt ${s.kind}`);
    }
  }

  cur = newBlock('entry');
  const tail = blockBody(f.body);
  if (!terminated()) emit({ op: 'ret', a: f.ret === 'unit' ? null : tail }, f.body);

  return rebuildCfg(fn);
}
