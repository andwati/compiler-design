import { C, irText, isBinop, OP_SYMBOL, R, rebuildCfg, uses } from './ir.js';
import type { IrFn, IrInstr, IrModule, Operand, RewriteEvent } from './types.js';

type Lat = number | 'NAC';
type State = Map<string, Lat>;

const NAC = 'NAC' as const; // "not a constant" in the propagation lattice

const evalOp = (op: string, a: number, b: number): number => {
  switch (op) {
    case 'add':
      return (a + b) | 0;
    case 'sub':
      return (a - b) | 0;
    case 'mul':
      return Math.imul(a, b);
    case 'lt':
      return a < b ? 1 : 0;
    case 'gt':
      return a > b ? 1 : 0;
    case 'eq':
      return a === b ? 1 : 0;
  }
  throw new Error(`evalOp ${op}`);
};

const val = (o: Operand, st: State): Lat => {
  if (o.k === 'const') return o.v;
  const v = st.get(o.name);
  return typeof v === 'number' ? v : NAC;
};

// Transfer function: effect of one instruction on the constant map.
function transfer(ins: IrInstr, st: State): void {
  if (!ins.dst) return;
  if (ins.op === 'copy') st.set(ins.dst, val(ins.a!, st));
  else if (isBinop(ins.op)) {
    const a = val(ins.a!, st),
      b = val(ins.b!, st);
    st.set(ins.dst, a !== NAC && b !== NAC ? evalOp(ins.op, a, b) : NAC);
  } else st.set(ins.dst, NAC); // call results are unknown
}

// Meet of two maps: equal constants survive, anything else becomes NAC. Missing = "no information yet".
function meet(into: State, from: State): boolean {
  let changed = false;
  for (const [k, v] of from) {
    if (!into.has(k)) {
      into.set(k, v);
      changed = true;
    } else if (into.get(k) !== v && into.get(k) !== NAC) {
      into.set(k, NAC);
      changed = true;
    }
  }
  return changed;
}

/** Forward dataflow to a fixpoint: constant state at the *entry* of every block. */
function constantsIn(fn: IrFn): Map<string, State> {
  const byLabel = new Map(fn.blocks.map((b) => [b.label, b]));
  const IN = new Map<string, State>(fn.blocks.map((b) => [b.label, new Map()]));
  for (const p of fn.params) IN.get(fn.blocks[0].label)!.set(p.name, NAC);
  const work = [fn.blocks[0].label];
  const visited = new Set<string>();
  while (work.length) {
    const l = work.shift()!;
    const st = new Map(IN.get(l)!);
    for (const ins of byLabel.get(l)!.instrs) transfer(ins, st);
    const first = !visited.has(l);
    visited.add(l);
    for (const s of byLabel.get(l)!.succ) {
      if (meet(IN.get(s)!, st) || (first && !visited.has(s))) work.push(s);
    }
  }
  return IN;
}

function propagate(fn: IrFn, round: number, events: RewriteEvent[]): boolean {
  const IN = constantsIn(fn);
  let changed = false;
  for (const b of fn.blocks) {
    const st: State = new Map(IN.get(b.label));
    for (const ins of b.instrs) {
      const before = irText(ins);
      const subs: string[] = [];
      const sub = (o: Operand): Operand => {
        const v = o.k === 'reg' ? st.get(o.name) : undefined;
        if (o.k === 'reg' && typeof v === 'number') {
          subs.push(`${o.name} = ${v}`);
          return C(v);
        }
        return o;
      };
      const orig = { ...ins };
      if (ins.a) ins.a = sub(ins.a);
      if (ins.b) ins.b = sub(ins.b);
      if (ins.args) ins.args = ins.args.map(sub);
      if (subs.length) {
        changed = true;
        ins.text = irText(ins);
        events.push({
          pass: 'constant-propagation',
          round,
          kind: 'propagate',
          fn: fn.name,
          block: b.label,
          instrId: ins.id,
          nodeId: ins.node,
          before,
          after: ins.text,
          reason: `${subs.join(', ')} on every path reaching here`,
        });
      }
      transfer(orig, st);
    }
  }
  return changed;
}

function fold(fn: IrFn, round: number, events: RewriteEvent[]): boolean {
  let changed = false;
  for (const b of fn.blocks) {
    for (const ins of b.instrs) {
      const before = irText(ins);
      if (isBinop(ins.op) && ins.a?.k === 'const' && ins.b?.k === 'const') {
        const v = evalOp(ins.op, ins.a.v, ins.b.v);
        const reason = `${ins.a.v} ${OP_SYMBOL[ins.op]} ${ins.b.v} = ${v} at compile time`;
        Object.assign(ins, { op: 'copy', a: C(v) });
        ins.b = undefined;
        ins.text = irText(ins);
        events.push({
          pass: 'constant-folding',
          round,
          kind: 'fold',
          fn: fn.name,
          block: b.label,
          instrId: ins.id,
          nodeId: ins.node,
          before,
          after: ins.text,
          reason,
        });
        changed = true;
      } else if (ins.op === 'br' && ins.a?.k === 'const') {
        const target = ins.a.v ? ins.then : ins.else;
        Object.assign(ins, { op: 'jmp', target });
        ins.a = undefined;
        ins.then = undefined;
        ins.else = undefined;
        ins.text = irText(ins);
        events.push({
          pass: 'constant-folding',
          round,
          kind: 'fold',
          fn: fn.name,
          block: b.label,
          instrId: ins.id,
          nodeId: ins.node,
          before,
          after: ins.text,
          reason: 'branch condition is a constant',
        });
        changed = true;
      }
    }
  }
  if (changed) rebuildCfg(fn);
  return changed;
}

function dce(fn: IrFn, round: number, events: RewriteEvent[]): boolean {
  // Backward liveness fixpoint.
  const liveOut = new Map<string, Set<string>>(fn.blocks.map((b) => [b.label, new Set()]));
  const liveIn = new Map<string, Set<string>>(fn.blocks.map((b) => [b.label, new Set()]));
  const step = (
    b: IrFn['blocks'][number],
    out: Set<string>,
    onIns?: (ins: IrInstr, i: number, live: Set<string>) => void,
  ) => {
    const live = new Set(out);
    for (let i = b.instrs.length - 1; i >= 0; i--) {
      const ins = b.instrs[i];
      if (onIns) onIns(ins, i, live);
      if (ins.dst) live.delete(ins.dst);
      for (const u of uses(ins)) live.add(u);
    }
    return live;
  };
  let again = true;
  while (again) {
    again = false;
    for (const b of [...fn.blocks].reverse()) {
      const out = liveOut.get(b.label)!;
      for (const s of b.succ) for (const v of liveIn.get(s)!) out.add(v);
      const inn = step(b, out);
      const old = liveIn.get(b.label)!;
      if (inn.size !== old.size || [...inn].some((v) => !old.has(v))) {
        liveIn.set(b.label, inn);
        again = true;
      }
    }
  }
  let changed = false;
  for (const b of fn.blocks) {
    const dead = new Set<number>();
    step(b, liveOut.get(b.label)!, (ins, i, live) => {
      if (!ins.dst || live.has(ins.dst)) return;
      const before = irText(ins);
      if (ins.op === 'call') {
        // A call has side effects: keep it, but drop its unused result.
        ins.dst = null;
        ins.text = irText(ins);
        events.push({
          pass: 'dead-code-elimination',
          round,
          kind: 'simplify',
          fn: fn.name,
          block: b.label,
          instrId: ins.id,
          nodeId: ins.node,
          before,
          after: ins.text,
          reason: 'result never used, call kept for its side effects',
        });
      } else {
        dead.add(i);
        events.push({
          pass: 'dead-code-elimination',
          round,
          kind: 'delete',
          fn: fn.name,
          block: b.label,
          instrId: ins.id,
          nodeId: ins.node,
          before,
          after: null,
          reason: `\`${ins.dst}\` is never read afterwards`,
        });
      }
      changed = true;
    });
    if (dead.size) b.instrs = b.instrs.filter((_, i) => !dead.has(i));
  }
  return changed;
}

/**
 * Optimizer: constant propagation -> constant folding -> dead code elimination, repeated until
 * nothing changes. Every rewrite is recorded as { before, after, reason } so the scene can play
 * them back; `snapshots` holds the IR after each round for the before/after views.
 */
export function optimize(irModule: IrModule): {
  ir: IrModule;
  events: RewriteEvent[];
  rounds: { fn: string; round: number; events: number }[];
} {
  const mod = structuredClone(irModule);
  const events: RewriteEvent[] = [];
  const rounds: { fn: string; round: number; events: number }[] = [];
  for (const fn of mod.functions) {
    let round = 0,
      changed = true;
    while (changed && round < 10) {
      round++;
      const n0 = events.length;
      const a = propagate(fn, round, events);
      const b = fold(fn, round, events);
      const c = dce(fn, round, events);
      changed = a || b || c;
      rounds.push({ fn: fn.name, round, events: events.length - n0 });
    }
    rebuildCfg(fn);
  }
  return { ir: mod, events, rounds };
}
export { R };
