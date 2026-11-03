import { isBinop, isCompare, uses } from './ir.js';
import type { IrFn, IrInstr, IrModule, Operand } from './types.js';
import { type AsmInstr, type AsmOperand, asmText, type Cond, imm, invertCc, mem, type Reg64, reg } from './x86.js';

export interface Regs {
  calleeSaved: Reg64[];
  callerSaved: Reg64[];
}
export interface Loc {
  kind: 'reg' | 'stack';
  reg?: Reg64;
  slot?: number;
}
export interface Interval {
  vreg: string;
  start: number;
  end: number;
  crossesCall: boolean;
  loc: Loc;
  spilled?: boolean;
}
export type AllocEvent =
  | { type: 'assign'; vreg: string; reg: Reg64; at: number }
  | { type: 'expire'; vreg: string; reg: Reg64; at: number }
  | { type: 'spill'; vreg: string; reg: Reg64 | null; takenBy: string | null; at: number };
export type AsmLine = AsmInstr & { irId: number | null; role: 'prologue' | 'body' | 'epilogue'; text: string };
export interface FnCode {
  name: string;
  asm: AsmLine[];
  intervals: Interval[];
  allocEvents: AllocEvent[];
  spills: string[];
  frame: { calleeSaved: Reg64[]; spillSlots: number; frameBytes: number };
  positions: { pos: number; irId: number; block: string; text?: string }[];
  callPositions: number[];
  fusedCompares: number[];
  regs: Regs;
}

// System V AMD64: integer args in rdi, rsi, rdx, rcx, r8, r9; result in eax.
const ARG_REGS: Reg64[] = ['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9'];
const CC_OF: Record<string, Cond> = { lt: 'l', gt: 'g', eq: 'e' };
const COMMUTATIVE = new Set(['add', 'mul']);

/**
 * Register file for the allocator. Deliberately disjoint from the ABI argument registers,
 * rax (result / scratch) and r11, which removes parallel-move hazards when setting up calls.
 *   calleeSaved: survive calls, so intervals live across a call MUST use these (or spill)
 *   callerSaved: free for short intervals, but clobbered by any call
 * Shrink `calleeSaved` to see linear scan spill (see tests).
 */
export const DEFAULT_REGS: Regs = { calleeSaved: ['rbx', 'r12', 'r13'], callerSaved: ['r10'] };

export function generate(irModule: IrModule, options: { regs?: Regs } = {}): { functions: FnCode[] } {
  const regs = options.regs ?? DEFAULT_REGS;
  return { functions: irModule.functions.map((fn) => genFn(fn, regs)) };
}

function genFn(fn: IrFn, regs: Regs): FnCode {
  // ---- 1. linearise: number every instruction, remember block boundaries ----
  const flat: { ins: IrInstr; block: string; pos: number }[] = [];
  const blockRange = new Map<string, [number, number]>();
  let pos = 1; // position 0 is the function entry where parameters are defined
  for (const b of fn.blocks) {
    const first = pos;
    for (const ins of b.instrs) flat.push({ ins, block: b.label, pos: pos++ });
    blockRange.set(b.label, [first, pos - 1]);
  }

  // ---- 2. liveness (backward dataflow) -> per-block live-in/out ----
  const liveIn = new Map<string, Set<string>>(fn.blocks.map((b) => [b.label, new Set()]));
  const liveOut = new Map<string, Set<string>>(fn.blocks.map((b) => [b.label, new Set()]));
  let again = true;
  while (again) {
    again = false;
    for (const b of [...fn.blocks].reverse()) {
      const out = liveOut.get(b.label)!;
      for (const s of b.succ) for (const v of liveIn.get(s)!) out.add(v);
      const live = new Set(out);
      for (let i = b.instrs.length - 1; i >= 0; i--) {
        const ins = b.instrs[i];
        if (ins.dst) live.delete(ins.dst);
        for (const u of uses(ins)) live.add(u);
      }
      const old = liveIn.get(b.label)!;
      if (live.size !== old.size || [...live].some((v) => !old.has(v))) {
        liveIn.set(b.label, live);
        again = true;
      }
    }
  }

  // ---- 3. compare/branch fusion: `t = lt a, b` immediately followed by the only use `br t` ----
  const useCount = new Map<string, number>();
  for (const { ins } of flat) for (const u of uses(ins)) useCount.set(u, (useCount.get(u) ?? 0) + 1);
  const fused = new Set<number>(); // instruction ids of fused compares
  const fusedVreg = new Set<string>();
  for (let i = 0; i < flat.length - 1; i++) {
    const a = flat[i].ins,
      b = flat[i + 1].ins;
    if (
      isCompare(a.op) &&
      b.op === 'br' &&
      b.a?.k === 'reg' &&
      b.a.name === a.dst &&
      useCount.get(a.dst!) === 1 &&
      flat[i].block === flat[i + 1].block
    ) {
      fused.add(a.id);
      fusedVreg.add(a.dst!);
    }
  }

  // ---- 4. live intervals: one [start, end] per vreg (Poletto & Sarkar linear scan) ----
  const iv = new Map<string, Interval>();
  const touch = (v: string, p: number) => {
    if (fusedVreg.has(v)) return;
    const r = iv.get(v);
    if (!r) iv.set(v, { vreg: v, start: p, end: p, crossesCall: false, loc: undefined as unknown as Loc });
    else {
      r.start = Math.min(r.start, p);
      r.end = Math.max(r.end, p);
    }
  };
  for (const p of fn.params) touch(p.name, 0);
  for (const { ins, pos: p } of flat) {
    if (ins.dst) touch(ins.dst, p);
    for (const u of uses(ins)) touch(u, p);
  }
  // A value live into / out of a block is live across the whole block: this is what stretches
  // `sum` and `i` over the entire loop even though their last textual use is earlier.
  for (const b of fn.blocks) {
    const [f, l] = blockRange.get(b.label)!;
    if (f > l) continue;
    for (const v of liveIn.get(b.label)!) touch(v, f);
    for (const v of liveOut.get(b.label)!) touch(v, l);
  }
  const callPositions = flat.filter(({ ins }) => ins.op === 'call').map(({ pos: p }) => p);
  const intervals = [...iv.values()].sort((a, b) => a.start - b.start || a.end - b.end);
  for (const it of intervals) it.crossesCall = callPositions.some((p) => it.start < p && it.end > p);

  // ---- 5. linear scan ----
  const free = new Set([...regs.callerSaved, ...regs.calleeSaved]);
  const calleeSet = new Set(regs.calleeSaved);
  const active: Interval[] = []; // intervals currently holding a register
  const events: AllocEvent[] = [];
  let nSlots = 0;
  const spill = (it: Interval) => {
    it.loc = { kind: 'stack', slot: nSlots++ };
    it.spilled = true;
  };
  for (const it of intervals) {
    // expire: intervals that ended at or before our start release their registers
    // (end == start is fine: an operand's last use and a destination's def may share a register)
    for (let k = active.length - 1; k >= 0; k--) {
      if (active[k].end <= it.start) {
        events.push({ type: 'expire', vreg: active[k].vreg, reg: active[k].loc.reg!, at: it.start });
        free.add(active[k].loc.reg!);
        active.splice(k, 1);
      }
    }
    const allowed = (r: Reg64) => !it.crossesCall || calleeSet.has(r);
    // prefer caller-saved registers for short intervals so callee-saved ones (which cost push/pop) stay free
    const order = [...regs.callerSaved, ...regs.calleeSaved];
    const pick = order.find((r) => free.has(r) && allowed(r));
    if (pick) {
      free.delete(pick);
      it.loc = { kind: 'reg', reg: pick };
      active.push(it);
      events.push({ type: 'assign', vreg: it.vreg, reg: pick, at: it.start });
    } else {
      // No register: spill whichever candidate lives longest (the new interval included).
      const cands = active.filter((a) => allowed(a.loc.reg!));
      const victim = cands.sort((a, b) => b.end - a.end)[0];
      if (victim && victim.end > it.end) {
        it.loc = { kind: 'reg', reg: victim.loc.reg! };
        events.push({ type: 'spill', vreg: victim.vreg, reg: victim.loc.reg!, takenBy: it.vreg, at: it.start });
        spill(victim);
        active.splice(active.indexOf(victim), 1);
        active.push(it);
        events.push({ type: 'assign', vreg: it.vreg, reg: it.loc.reg!, at: it.start });
      } else {
        spill(it);
        events.push({ type: 'spill', vreg: it.vreg, reg: null, takenBy: null, at: it.start });
      }
    }
  }
  const locOf = new Map(intervals.map((i) => [i.vreg, i.loc]));

  // ---- 6. frame layout ----
  const usedCallee = regs.calleeSaved.filter((r) => intervals.some((i) => i.loc.kind === 'reg' && i.loc.reg! === r));
  const savedBytes = usedCallee.length * 8;
  let frame = nSlots * 8;
  if ((savedBytes + frame) % 16) frame += 8; // keep rsp 16-byte aligned at call sites (after push rbp it is aligned)
  const slotDisp = (slot: number) => -(savedBytes + 8 * (slot + 1));

  const asm: AsmLine[] = [];
  let curIr: number | null = null;
  let role: AsmLine['role'] = 'body';
  const out = (ins: AsmInstr) => {
    asm.push({ ...ins, irId: curIr, role, text: asmText(ins) });
  };
  const lblName = (l: string) => `.L${fn.name}_${l}`;

  const place = (v: Operand): AsmOperand => {
    if (v.k === 'const') return imm(v.v);
    const l = locOf.get(v.name)!;
    return l.kind === 'reg' ? reg(l.reg!) : mem(slotDisp(l.slot!));
  };
  const same = (a: AsmOperand, b: AsmOperand) =>
    a.k === b.k &&
    (a.k === 'reg'
      ? a.r === (b as typeof a).r
      : a.k === 'mem'
        ? a.disp === (b as typeof a).disp
        : a.v === (b as typeof a).v);
  const mov = (d: AsmOperand, s: AsmOperand, size: 32 | 64 = 32) => {
    if (d.k === 'reg' && s.k === 'reg' && d.r === s.r) return;
    out({ op: 'mov', size, dst: d, src: s });
  };
  // mov that is legal for any (reg|mem|imm) -> (reg|mem) pair except mem->mem
  const movSafe = (d: AsmOperand, s: AsmOperand) => {
    if (d.k === 'mem' && s.k === 'mem') {
      mov(reg('rax'), s);
      mov(d, reg('rax'));
    } else mov(d, s);
  };
  const EAX = reg('rax');

  // ---- 7. instruction selection ----
  role = 'prologue';
  out({ op: 'label', name: fn.name, global: true });
  out({ op: 'push', size: 64, dst: reg('rbp') });
  out({ op: 'mov', size: 64, dst: reg('rbp'), src: reg('rsp') });
  for (const r of usedCallee) out({ op: 'push', size: 64, dst: reg(r) });
  if (frame) out({ op: 'sub', size: 64, dst: reg('rsp'), src: imm(frame) });
  fn.params.forEach((p, i: number) => {
    curIr = null;
    const l = locOf.get(p.name);
    if (l) movSafe(l.kind === 'reg' ? reg(l.reg!) : mem(slotDisp(l.slot!)), reg(ARG_REGS[i]));
  });
  role = 'body';

  const epilogue = () => {
    role = 'epilogue';
    if (frame) out({ op: 'add', size: 64, dst: reg('rsp'), src: imm(frame) });
    for (const r of [...usedCallee].reverse()) out({ op: 'pop', size: 64, dst: reg(r) });
    out({ op: 'pop', size: 64, dst: reg('rbp') });
    out({ op: 'ret' });
    role = 'body';
  };

  let pendingCc: Cond | null = null; // condition produced by a fused compare
  fn.blocks.forEach((b, bi: number) => {
    const nextLabel = fn.blocks[bi + 1]?.label;
    curIr = null;
    out({ op: 'label', name: lblName(b.label), block: b.label });
    for (const ins of b.instrs) {
      curIr = ins.id;
      if (isBinop(ins.op) && !fused.has(ins.id)) selectBinop(ins);
      else if (fused.has(ins.id)) {
        selectCompareOperands(ins);
        pendingCc = CC_OF[ins.op];
      } else if (ins.op === 'copy') {
        if (locOf.has(ins.dst!)) movSafe(place({ k: 'reg', name: ins.dst! }), place(ins.a!));
      } else if (ins.op === 'call') {
        ins.args!.forEach((a, i: number) => {
          mov(reg(ARG_REGS[i]), place(a));
        });
        out({ op: 'call', sym: ins.callee! });
        if (ins.dst && locOf.has(ins.dst)) movSafe(place({ k: 'reg', name: ins.dst }), EAX);
      } else if (ins.op === 'ret') {
        if (ins.a) mov(EAX, place(ins.a));
        epilogue();
      } else if (ins.op === 'jmp') {
        if (ins.target !== nextLabel) out({ op: 'jmp', target: lblName(ins.target!) });
      } else if (ins.op === 'br') {
        let cc: Cond;
        if (pendingCc) {
          cc = pendingCc;
          pendingCc = null;
        } else if (ins.a!.k === 'const') {
          const t = ins.a!.v ? ins.then! : ins.else!;
          if (t !== nextLabel) out({ op: 'jmp', target: lblName(t) });
          continue;
        } else {
          out({ op: 'cmp', size: 32, dst: place(ins.a!), src: imm(0) });
          cc = 'ne';
        }
        if (ins.else === nextLabel) out({ op: 'jcc', cond: cc, target: lblName(ins.then!) });
        else if (ins.then === nextLabel) out({ op: 'jcc', cond: invertCc(cc), target: lblName(ins.else!) });
        else {
          out({ op: 'jcc', cond: cc, target: lblName(ins.then!) });
          out({ op: 'jmp', target: lblName(ins.else!) });
        }
      } else throw new Error(`codegen: unhandled ${ins.op}`);
    }
  });

  // cmp a, b needs a register-or-memory left side and (reg|mem|imm) right side, but not mem,mem or imm,x
  function selectCompareOperands(ins: IrInstr) {
    let a = place(ins.a!);
    const b = place(ins.b!);
    // the left side must be a register, or memory with a non-memory right side
    if (a.k === 'imm' || (a.k === 'mem' && b.k === 'mem')) {
      mov(EAX, a);
      a = EAX;
    }
    out({ op: 'cmp', size: 32, dst: a, src: b });
  }

  function selectBinop(ins: IrInstr) {
    if (isCompare(ins.op)) {
      selectCompareOperands(ins);
      out({ op: 'setcc', cond: CC_OF[ins.op], dst: EAX });
      const d = place({ k: 'reg', name: ins.dst! });
      if (d.k === 'reg') out({ op: 'movzx', dst: d, src: EAX });
      else {
        out({ op: 'movzx', dst: EAX, src: EAX });
        mov(d, EAX);
      }
      return;
    }
    let a = place(ins.a!),
      b = place(ins.b!);
    const d = place({ k: 'reg', name: ins.dst! });
    // constant on the left of a commutative op: swap so the immediate is the right operand
    if (a.k === 'imm' && b.k !== 'imm' && COMMUTATIVE.has(ins.op)) [a, b] = [b, a];
    if (d.k === 'reg' && !same(d, b)) {
      work(d);
    } else if (d.k === 'reg' && COMMUTATIVE.has(ins.op) && !same(d, a)) {
      [a, b] = [b, a];
      work(d);
    } else if (d.k === 'reg' && same(d, a) && same(d, b)) {
      work(d);
    } else {
      work(EAX);
      movSafe(d, EAX);
    }
    // w = a (op) b, computed in register w
    function work(w: AsmOperand) {
      if (ins.op === 'mul') {
        if (b.k === 'imm') {
          if (a.k === 'imm') mov(w, a);
          out({ op: 'imul3', dst: w, src: a.k === 'imm' ? w : a, imm: b.v });
          return;
        }
        mov(w, a);
        out({ op: 'imul', dst: w, src: b });
      } else {
        mov(w, a);
        out({ op: ins.op as 'add' | 'sub', size: 32, dst: w, src: b });
      }
    }
  }

  return {
    name: fn.name,
    asm,
    intervals,
    allocEvents: events,
    spills: intervals.filter((i) => i.spilled).map((i) => i.vreg),
    frame: { calleeSaved: usedCallee, spillSlots: nSlots, frameBytes: frame },
    positions: flat.map(({ ins, block, pos: p }) => ({ pos: p, irId: ins.id, block, text: ins.text })),
    callPositions,
    fusedCompares: [...fused],
    regs,
  };
}
