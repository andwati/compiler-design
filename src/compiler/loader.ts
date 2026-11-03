import { PF, PT, parseElf } from './elf.js';
import { REG64, type Reg64 } from './x86.js';

export const STACK_TOP = 0x7ffffffff000;
export const STACK_SIZE = 0x21000; // 132 KiB, like Linux's initial stack mapping

export interface Region {
  name: string;
  vaddr: number;
  size: number;
  flags: number;
  data: Uint8Array;
}

/** A virtual address space made of mapped regions. Accesses are permission-checked like an MMU would. */
export class AddressSpace {
  regions: Region[] = [];
  map(r: Region) {
    this.regions.push(r);
  }
  private find(addr: number, size: number, need: number): { r: Region; off: number } {
    for (const r of this.regions) {
      if (addr >= r.vaddr && addr + size <= r.vaddr + r.size) {
        if (!(r.flags & need))
          throw new Fault(
            `protection fault at ${addr.toString(16)} (${need === PF.W ? 'write' : need === PF.X ? 'execute' : 'read'})`,
          );
        return { r, off: addr - r.vaddr };
      }
    }
    throw new Fault(`segmentation fault at ${addr.toString(16)}`);
  }
  read8(a: number, need: number = PF.R) {
    const { r, off } = this.find(a, 1, need);
    return r.data[off];
  }
  read32(a: number) {
    const { r, off } = this.find(a, 4, PF.R);
    return (r.data[off] | (r.data[off + 1] << 8) | (r.data[off + 2] << 16) | (r.data[off + 3] << 24)) >>> 0;
  }
  read64(a: number) {
    return this.read32(a) + this.read32(a + 4) * 2 ** 32;
  }
  write32(a: number, v: number) {
    const { r, off } = this.find(a, 4, PF.W);
    for (let i = 0; i < 4; i++) r.data[off + i] = (v >>> (8 * i)) & 0xff;
  }
  write64(a: number, v: number) {
    this.write32(a, v % 2 ** 32);
    this.write32(a + 4, Math.floor(v / 2 ** 32));
  }
  fetch8(a: number) {
    return this.read8(a, PF.X);
  }
}
export class Fault extends Error {}

export interface Flags {
  ZF: number;
  SF: number;
  CF: number;
  OF: number;
}
export interface CpuState {
  regs: number[];
  rip: number;
  flags: Flags;
}

export interface LoadResult {
  mem: AddressSpace;
  cpu: CpuState;
  entry: number;
  /** what the kernel did, in order, for the scene */
  events: (
    | { type: 'map'; name: string; vaddr: number; size: number; flags: number; fileOffset: number }
    | { type: 'stack'; top: number; size: number; rsp: number }
    | { type: 'start'; rip: number; rsp: number }
  )[];
}

/** Simulate execve(): map PT_LOAD segments, build the initial stack, point rip/rsp. */
export function load(file: number[]): LoadResult {
  const elf = parseElf(file);
  if (elf.header.type !== 2) throw new Error('loader: not an executable (ET_EXEC)');
  const mem = new AddressSpace();
  const events: LoadResult['events'] = [];
  for (const ph of elf.programHeaders) {
    if (ph.type !== PT.LOAD) continue;
    const data = new Uint8Array(ph.memsz); // memsz > filesz tail stays zero (.bss semantics)
    data.set(file.slice(ph.offset, ph.offset + ph.filesz));
    const name = ph.flags & PF.X ? 'text' : 'headers';
    mem.map({ name, vaddr: ph.vaddr, size: ph.memsz, flags: ph.flags, data });
    events.push({ type: 'map', name, vaddr: ph.vaddr, size: ph.memsz, flags: ph.flags, fileOffset: ph.offset });
  }
  const stack = new Uint8Array(STACK_SIZE);
  mem.map({ name: 'stack', vaddr: STACK_TOP - STACK_SIZE, size: STACK_SIZE, flags: PF.R | PF.W, data: stack });
  // Initial stack as the kernel leaves it: argc = 0, argv[] = NULL, envp[] = NULL, auxv end. rsp points at argc.
  const rsp = STACK_TOP - 0x20;
  const regs = new Array(16).fill(0);
  regs[REG64.indexOf('rsp')] = rsp;
  events.push({ type: 'stack', top: STACK_TOP, size: STACK_SIZE, rsp });
  events.push({ type: 'start', rip: elf.header.entry, rsp });
  return {
    mem,
    cpu: { regs, rip: elf.header.entry, flags: { ZF: 0, SF: 0, CF: 0, OF: 0 } },
    entry: elf.header.entry,
    events,
  };
}

export interface Step {
  n: number;
  rip: number;
  len: number;
  bytes: number[];
  text: string;
  kind: 'call' | 'ret' | 'jump' | 'syscall' | 'other';
  regs: number[]; // snapshot AFTER the instruction (indexed like REG64)
  changed: Reg64[];
  flags: Flags;
  nextRip: number;
  writes: { addr: number; size: number; value: number }[];
  reads: { addr: number; size: number }[];
}
export interface RunResult {
  steps: Step[];
  exitCode: number;
}

const R = (n: number) => REG64[n];
const r32 = [
  'eax',
  'ecx',
  'edx',
  'ebx',
  'esp',
  'ebp',
  'esi',
  'edi',
  'r8d',
  'r9d',
  'r10d',
  'r11d',
  'r12d',
  'r13d',
  'r14d',
  'r15d',
];
const COND = ['o', 'no', 'b', 'ae', 'e', 'ne', 'be', 'a', 's', 'ns', 'p', 'np', 'l', 'ge', 'le', 'g'];
const sx8 = (v: number) => (v << 24) >> 24;
const hexs = (v: number) => (v < 0 ? `-0x${(-v).toString(16)}` : `0x${v.toString(16)}`);

function condHolds(cc: number, f: Flags): boolean {
  switch (cc) {
    case 0x4:
      return f.ZF === 1;
    case 0x5:
      return f.ZF === 0;
    case 0xc:
      return f.SF !== f.OF;
    case 0xd:
      return f.SF === f.OF;
    case 0xe:
      return f.ZF === 1 || f.SF !== f.OF;
    case 0xf:
      return f.ZF === 0 && f.SF === f.OF;
    default:
      throw new Fault(`unsupported condition code ${cc}`);
  }
}

/**
 * Interpreter for exactly the opcodes our codegen emits (plus syscall/exit). It decodes from the bytes
 * in the *loaded memory image*, not from the assembler's tables, so it genuinely executes the linked file.
 * 64-bit registers are JS numbers: safe because the only 64-bit values ever held are addresses (< 2^47).
 */
export function run(ld: LoadResult, maxSteps = 100000): RunResult {
  const { mem, cpu } = ld;
  const steps: Step[] = [];
  const RSP = 4;

  for (let n = 0; n < maxSteps; n++) {
    const start = cpu.rip;
    const before = cpu.regs.slice();
    const writes: Step['writes'] = [];
    const reads: Step['reads'] = [];
    let p = start;
    const next8 = () => mem.fetch8(p++);
    const next32 = () => {
      let v = 0;
      for (let i = 0; i < 4; i++) v += next8() * 2 ** (8 * i);
      return v | 0;
    };

    let rex = 0;
    let op = next8();
    if ((op & 0xf0) === 0x40) {
      rex = op;
      op = next8();
    }
    const W = (rex >> 3) & 1,
      Rx = ((rex >> 2) & 1) << 3,
      Bx = (rex & 1) << 3;
    const name = (n: number, w = W) => (w ? R(n) : r32[n]);
    const getReg = (n: number, w = W) => (w ? cpu.regs[n] : cpu.regs[n] % 2 ** 32);
    const setReg = (n: number, v: number, w = W) => {
      cpu.regs[n] = w ? v : v >>> 0;
    };

    // ModRM decoding: returns accessors + display text for the r/m operand
    const modrm = () => {
      const m = next8();
      const mod = m >> 6,
        reg = ((m >> 3) & 7) | Rx;
      const rmLow = m & 7,
        rm = rmLow | Bx;
      if (mod === 3) {
        return { reg, isReg: true, rm, text: name(rm), get: () => getReg(rm), set: (v: number) => setReg(rm, v) };
      }
      if (rmLow === 4 || (mod === 0 && rmLow === 5))
        throw new Fault('SIB / rip-relative addressing is not supported by this interpreter');
      const disp = mod === 1 ? sx8(next8()) : mod === 2 ? next32() : 0;
      const ea = () => cpu.regs[rm] + disp;
      return {
        reg,
        isReg: false,
        rm,
        text: `[${R(rm)}${disp < 0 ? '-' : '+'}${Math.abs(disp)}]`,
        get: () => {
          const a = ea();
          reads.push({ addr: a, size: 4 });
          return mem.read32(a);
        },
        set: (v: number) => {
          const a = ea();
          mem.write32(a, v);
          writes.push({ addr: a, size: 4, value: v >>> 0 });
        },
      };
    };
    const setFlagsSub = (a: number, b: number) => {
      const res = (a - b) >>> 0;
      cpu.flags = { ZF: +(res === 0), SF: res >>> 31, CF: +(a >>> 0 < b >>> 0), OF: ((a ^ b) & (a ^ res)) >>> 31 };
      return res;
    };
    const setFlagsAdd = (a: number, b: number) => {
      const res = (a + b) >>> 0;
      cpu.flags = { ZF: +(res === 0), SF: res >>> 31, CF: +(a + b > 0xffffffff), OF: (~(a ^ b) & (a ^ res)) >>> 31 };
      return res;
    };

    let text = '';
    let kind: Step['kind'] = 'other';
    let nextRip = 0;
    let exit: number | null = null;
    const finishAt = () => p;

    if (op >= 0x50 && op <= 0x5f) {
      const r = (op & 7) | Bx;
      if (op < 0x58) {
        cpu.regs[RSP] -= 8;
        mem.write64(cpu.regs[RSP], cpu.regs[r]);
        writes.push({ addr: cpu.regs[RSP], size: 8, value: cpu.regs[r] });
        text = `push ${R(r)}`;
      } else {
        reads.push({ addr: cpu.regs[RSP], size: 8 });
        cpu.regs[r] = mem.read64(cpu.regs[RSP]);
        cpu.regs[RSP] += 8;
        text = `pop ${R(r)}`;
      }
    } else if (op === 0x89 || op === 0x8b) {
      const m = modrm();
      if (op === 0x8b) {
        setReg(m.reg, m.get());
        text = `mov ${name(m.reg)}, ${m.text}`;
      } else {
        m.set(getReg(m.reg));
        text = `mov ${m.text}, ${name(m.reg)}`;
      }
    } else if (op >= 0xb8 && op <= 0xbf) {
      const r = (op & 7) | Bx;
      const v = next32();
      setReg(r, v, 0);
      text = `mov ${r32[r]}, ${v}`;
    } else if (op === 0xc7) {
      const m = modrm();
      const v = next32();
      m.set(v >>> 0);
      text = `mov dword ${m.text}, ${v}`;
    } else if (op === 0x03 || op === 0x2b || op === 0x3b) {
      const m = modrm();
      const a = getReg(m.reg, 0),
        b = m.get();
      const mn = op === 0x03 ? 'add' : op === 0x2b ? 'sub' : 'cmp';
      const res = op === 0x03 ? setFlagsAdd(a, b) : setFlagsSub(a, b);
      if (mn !== 'cmp') setReg(m.reg, res, 0);
      text = `${mn} ${name(m.reg, 0)}, ${m.text}`;
    } else if (op === 0x81 || op === 0x83) {
      const m = modrm();
      const ext = m.reg & 7;
      const imm = op === 0x83 ? sx8(next8()) : next32();
      const mn = ({ 0: 'add', 5: 'sub', 7: 'cmp' } as Record<number, string>)[ext];
      if (!mn) throw new Fault(`unsupported 0x${op.toString(16)} /${ext}`);
      if (W) {
        // 64-bit stack-pointer arithmetic: flags are not modelled for 64-bit forms
        const cur = cpu.regs[m.rm];
        if (mn === 'add') cpu.regs[m.rm] = cur + imm;
        else if (mn === 'sub') cpu.regs[m.rm] = cur - imm;
        text = `${mn} ${m.text}, ${imm}`;
      } else {
        const a = m.get(),
          b = imm >>> 0;
        const res = mn === 'add' ? setFlagsAdd(a, b) : setFlagsSub(a, b);
        if (mn !== 'cmp') m.set(res);
        text = `${mn} ${m.text}, ${imm}`;
      }
    } else if (op === 0x6b || op === 0x69) {
      const m = modrm();
      const imm = op === 0x6b ? sx8(next8()) : next32();
      const res = Math.imul(m.get(), imm);
      setReg(m.reg, res, 0);
      text = `imul ${r32[m.reg]}, ${m.text}, ${imm}`;
    } else if (op === 0xe8 || op === 0xe9) {
      const rel = next32();
      const target = p + rel;
      if (op === 0xe8) {
        cpu.regs[RSP] -= 8;
        mem.write64(cpu.regs[RSP], p);
        writes.push({ addr: cpu.regs[RSP], size: 8, value: p });
        kind = 'call';
        text = `call ${hexs(target)}`;
      } else {
        kind = 'jump';
        text = `jmp ${hexs(target)}`;
      }
      nextRip = target;
    } else if (op === 0xc3) {
      reads.push({ addr: cpu.regs[RSP], size: 8 });
      nextRip = mem.read64(cpu.regs[RSP]);
      cpu.regs[RSP] += 8;
      kind = 'ret';
      text = 'ret';
    } else if (op === 0x0f) {
      const op2 = next8();
      if (op2 === 0x05) {
        kind = 'syscall';
        text = 'syscall';
        const rax = cpu.regs[0];
        if (rax !== 60) throw new Fault(`unsupported syscall ${rax}`);
        exit = cpu.regs[7] & 0xff;
      } else if (op2 >= 0x80 && op2 <= 0x8f) {
        const rel = next32();
        const target = p + rel;
        kind = 'jump';
        text = `j${COND[op2 & 15]} ${hexs(target)}`;
        nextRip = condHolds(op2 & 15, cpu.flags) ? target : p;
      } else if (op2 === 0xaf) {
        const m = modrm();
        setReg(m.reg, Math.imul(getReg(m.reg, 0), m.get()), 0);
        text = `imul ${r32[m.reg]}, ${m.text}`;
      } else if (op2 >= 0x90 && op2 <= 0x9f) {
        const m = modrm();
        const v = condHolds(op2 & 15, cpu.flags) ? 1 : 0;
        cpu.regs[m.rm] = (cpu.regs[m.rm] & ~0xff) | v;
        text = `set${COND[op2 & 15]} al`;
      } else if (op2 === 0xb6) {
        const m = modrm();
        setReg(m.reg, cpu.regs[m.rm] & 0xff, 0);
        text = `movzx ${r32[m.reg]}, al`;
      } else throw new Fault(`unsupported opcode 0f ${op2.toString(16)} at ${start.toString(16)}`);
    } else throw new Fault(`unsupported opcode ${op.toString(16)} at ${start.toString(16)}`);

    const len = finishAt() - start;
    if (kind === 'other' || kind === 'syscall') nextRip = p;
    cpu.rip = nextRip;
    const bytesOut = Array.from({ length: len }, (_, i) => mem.fetch8(start + i));
    const changed = REG64.filter((_, i) => cpu.regs[i] !== before[i]);
    steps.push({
      n,
      rip: start,
      len,
      bytes: bytesOut,
      text,
      kind,
      regs: cpu.regs.slice(),
      changed,
      flags: { ...cpu.flags },
      nextRip: cpu.rip,
      writes,
      reads,
    });
    if (exit !== null) return { steps, exitCode: exit };
  }
  throw new Fault(`exceeded ${maxSteps} steps`);
}
