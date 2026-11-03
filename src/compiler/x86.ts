/**
 * Real x86-64 machine-code encoder for exactly the instruction forms our codegen emits.
 *
 * Instruction shape:  { op, size?, dst?, src?, imm?, cond?, target?, sym? }
 * Operands:           { k:'reg', r:'rbx' } | { k:'mem', base:'rbp', disp } | { k:'imm', v }
 * Registers are named by their 64-bit name; `size` (32|64) picks the view. REX.W selects 64-bit.
 */
export type Reg64 =
  | 'rax'
  | 'rcx'
  | 'rdx'
  | 'rbx'
  | 'rsp'
  | 'rbp'
  | 'rsi'
  | 'rdi'
  | 'r8'
  | 'r9'
  | 'r10'
  | 'r11'
  | 'r12'
  | 'r13'
  | 'r14'
  | 'r15';
export type Cond = 'e' | 'ne' | 'l' | 'ge' | 'le' | 'g';
export type AsmOperand = { k: 'reg'; r: Reg64 } | { k: 'mem'; base: Reg64; disp: number } | { k: 'imm'; v: number };
export interface AsmInstr {
  op:
    | 'label'
    | 'push'
    | 'pop'
    | 'ret'
    | 'syscall'
    | 'mov'
    | 'add'
    | 'sub'
    | 'cmp'
    | 'imul'
    | 'imul3'
    | 'setcc'
    | 'movzx'
    | 'jmp'
    | 'jcc'
    | 'call';
  size?: 32 | 64;
  dst?: AsmOperand;
  src?: AsmOperand;
  imm?: number;
  cond?: Cond;
  target?: string;
  sym?: string;
  name?: string;
  global?: boolean;
  block?: string;
}
export type SegKind = 'rex' | 'opcode' | 'modrm' | 'disp' | 'imm' | 'rel';
export interface Encoded {
  segments: { kind: SegKind; bytes: number[] }[];
  bytes: number[];
  fixup: { label?: string; sym?: string } | null;
}

export const REG64: Reg64[] = [
  'rax',
  'rcx',
  'rdx',
  'rbx',
  'rsp',
  'rbp',
  'rsi',
  'rdi',
  'r8',
  'r9',
  'r10',
  'r11',
  'r12',
  'r13',
  'r14',
  'r15',
];
const REG32 = [
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
export const regNum = (r: Reg64) => REG64.indexOf(r);
export const regName = (r: Reg64, size: 8 | 32 | 64 = 64) =>
  size === 32 ? REG32[regNum(r)] : size === 8 ? (regNum(r) === 0 ? 'al' : `${r}b`) : r;

export const reg = (r: Reg64): AsmOperand => ({ k: 'reg', r });
export const mem = (disp: number, base: Reg64 = 'rbp'): AsmOperand => ({ k: 'mem', base, disp });
export const imm = (v: number): AsmOperand => ({ k: 'imm', v });

// Condition codes: the low nibble of Jcc (0F 8x) and SETcc (0F 9x).
export const CC: Record<Cond, number> = { e: 0x4, ne: 0x5, l: 0xc, ge: 0xd, le: 0xe, g: 0xf };
export const INVERT_CC: Record<Cond, Cond> = { e: 'ne', ne: 'e', l: 'ge', ge: 'l', g: 'le', le: 'g' };
export const invertCc = (c: Cond): Cond => INVERT_CC[c];

const fits8 = (v: number) => v >= -128 && v <= 127;
export const le32 = (v: number) => [v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, (v >> 24) & 0xff];

// /digit for the 0x81/0x83 immediate group, and reg,r/m opcodes for the two-operand ALU ops.
const ALU: Record<string, { ext: number; rm: number }> = {
  add: { ext: 0, rm: 0x03 },
  sub: { ext: 5, rm: 0x2b },
  cmp: { ext: 7, rm: 0x3b },
};

/** Build ModRM (+disp) for `regField` and an r/m operand; returns { rex bits, bytes segments }. */
const regOf = (o: AsmOperand): Reg64 => {
  if (o.k !== 'reg') throw new Error('x86: expected a register operand');
  return o.r;
};

function modrm(regField: number, rm: AsmOperand): { R: number; B: number; segs: [SegKind, number[]][] } {
  const R = (regField >> 3) & 1;
  if (rm.k === 'reg') {
    const n = regNum(rm.r);
    return { R, B: (n >> 3) & 1, segs: [['modrm', [0xc0 | ((regField & 7) << 3) | (n & 7)]]] };
  }
  if (rm.k === 'mem') {
    const n = regNum(rm.base);
    if ((n & 7) === 4) throw new Error('x86: rsp/r12 base needs SIB, not supported by this encoder');
    const d8 = fits8(rm.disp);
    const mod = d8 ? 0x40 : 0x80; // always disp8/disp32: base rbp/r13 has no disp-less form
    return {
      R,
      B: (n >> 3) & 1,
      segs: [
        ['modrm', [mod | ((regField & 7) << 3) | (n & 7)]],
        ['disp', d8 ? [rm.disp & 0xff] : le32(rm.disp)],
      ],
    };
  }
  throw new Error('x86: bad r/m operand');
}

/**
 * Encode one instruction. Returns segments (prefix / opcode / modrm / disp / imm / rel) so the scene can
 * colour each byte by its role. Branch/call displacements are emitted as zero placeholders and flagged
 * with `fixup` for the assembler to patch (labels) or turn into relocations (symbols).
 */
export function encode(ins: AsmInstr): Encoded {
  const segs: [SegKind, number[]][] = [];
  let fixup: Encoded['fixup'] = null;
  const size = ins.size ?? 32;

  // Assemble: [REX] opcode(s) [modrm disp] [imm]
  const emit = ({
    op,
    regField = 0,
    rm = null,
    W = size === 64 ? 1 : 0,
    immBytes = null,
    forceRex = false,
    plusReg = null,
  }: {
    op: number[];
    regField?: number;
    rm?: AsmOperand | null;
    W?: number;
    immBytes?: number[] | null;
    forceRex?: boolean;
    plusReg?: number | null;
  }) => {
    let R = 0,
      B = 0,
      tail: [SegKind, number[]][] = [];
    if (rm) {
      const m = modrm(regField, rm);
      R = m.R;
      B = m.B;
      tail = m.segs;
    }
    if (plusReg !== null) B = (plusReg >> 3) & 1;
    if (W || R || B || forceRex) segs.push(['rex', [0x40 | (W << 3) | (R << 2) | B]]);
    segs.push(['opcode', op]);
    segs.push(...tail);
    if (immBytes) segs.push(['imm', immBytes]);
  };

  switch (ins.op) {
    case 'push':
    case 'pop': {
      const n = regNum(regOf(ins.dst!));
      emit({ op: [(ins.op === 'push' ? 0x50 : 0x58) | (n & 7)], W: 0, plusReg: n });
      break;
    }
    case 'ret':
      emit({ op: [0xc3], W: 0 });
      break;
    case 'syscall':
      emit({ op: [0x0f, 0x05], W: 0 });
      break;
    case 'mov': {
      const dst = ins.dst!,
        src = ins.src!;
      if (src.k === 'imm' && dst.k === 'reg') {
        const n = regNum(dst.r);
        emit({ op: [0xb8 | (n & 7)], plusReg: n, W: 0, immBytes: le32(src.v) });
      } else if (src.k === 'imm') {
        emit({ op: [0xc7], regField: 0, rm: dst, immBytes: le32(src.v) });
      } else if (dst.k === 'reg') {
        emit({ op: [0x8b], regField: regNum(dst.r), rm: src }); // mov r, r/m
      } else {
        emit({ op: [0x89], regField: regNum(regOf(src)), rm: dst }); // mov r/m, r
      }
      break;
    }
    case 'add':
    case 'sub':
    case 'cmp': {
      const dst = ins.dst!,
        src = ins.src!;
      const a = ALU[ins.op];
      if (src.k === 'imm') {
        const s8 = fits8(src.v);
        emit({ op: [s8 ? 0x83 : 0x81], regField: a.ext, rm: dst, immBytes: s8 ? [src.v & 0xff] : le32(src.v) });
      } else {
        if (dst.k !== 'reg') throw new Error(`x86: ${ins.op} needs a register destination`);
        emit({ op: [a.rm], regField: regNum(dst.r), rm: src });
      }
      break;
    }
    case 'imul':
      emit({ op: [0x0f, 0xaf], regField: regNum(regOf(ins.dst!)), rm: ins.src! });
      break;
    case 'imul3': {
      // imul r32, r/m32, imm
      const s8 = fits8(ins.imm!);
      emit({
        op: [s8 ? 0x6b : 0x69],
        regField: regNum(regOf(ins.dst!)),
        rm: ins.src!,
        immBytes: s8 ? [ins.imm! & 0xff] : le32(ins.imm!),
      });
      break;
    }
    case 'setcc':
      emit({ op: [0x0f, 0x90 | CC[ins.cond!]], regField: 0, rm: ins.dst!, W: 0 });
      break;
    case 'movzx':
      emit({ op: [0x0f, 0xb6], regField: regNum(regOf(ins.dst!)), rm: ins.src!, W: 0 });
      break; // src is the 8-bit view
    case 'jmp':
      emit({ op: [0xe9], W: 0, immBytes: [0, 0, 0, 0] });
      fixup = { label: ins.target };
      break;
    case 'jcc':
      emit({ op: [0x0f, 0x80 | CC[ins.cond!]], W: 0, immBytes: [0, 0, 0, 0] });
      fixup = { label: ins.target };
      break;
    case 'call':
      emit({ op: [0xe8], W: 0, immBytes: [0, 0, 0, 0] });
      fixup = { sym: ins.sym };
      break;
    default:
      throw new Error(`x86: cannot encode '${ins.op}'`);
  }
  // Retag the displacement of branches as 'rel' for display.
  if (fixup) segs[segs.length - 1][0] = 'rel';
  const bytes = segs.flatMap(([, b]) => b);
  return { segments: segs.map(([kind, b]) => ({ kind, bytes: b })), bytes, fixup };
}

const hex = (v: number) => (v < 0 ? `-0x${(-v).toString(16)}` : `0x${v.toString(16)}`);
const fmtOpnd = (o: AsmOperand, size: 8 | 32 | 64) => {
  if (o.k === 'reg') return regName(o.r, size);
  if (o.k === 'imm') return String(o.v);
  return `[${o.base}${o.disp < 0 ? '-' : '+'}${Math.abs(o.disp)}]`;
};

/** Intel-syntax text for display. */
export function asmText(ins: AsmInstr): string {
  const s = ins.size ?? 32;
  switch (ins.op) {
    case 'label':
      return `${ins.name}:`;
    case 'ret':
    case 'syscall':
      return ins.op;
    case 'push':
    case 'pop':
      return `${ins.op} ${regName(regOf(ins.dst!))}`;
    case 'jmp':
      return `jmp ${ins.target}`;
    case 'jcc':
      return `j${ins.cond} ${ins.target}`;
    case 'call':
      return `call ${ins.sym}`;
    case 'setcc':
      return `set${ins.cond} al`;
    case 'movzx':
      return `movzx ${regName(regOf(ins.dst!), 32)}, al`;
    case 'imul3':
      return `imul ${fmtOpnd(ins.dst!, 32)}, ${fmtOpnd(ins.src!, 32)}, ${ins.imm!}`;
    case 'mov':
      if (ins.dst!.k === 'mem' && ins.src!.k === 'imm') return `mov dword ${fmtOpnd(ins.dst!, s)}, ${ins.src!.v}`;
      return `mov ${fmtOpnd(ins.dst!, s)}, ${fmtOpnd(ins.src!, s)}`;
    default:
      return `${ins.op} ${fmtOpnd(ins.dst!, s)}, ${fmtOpnd(ins.src!, s)}`;
  }
}
export { hex };
