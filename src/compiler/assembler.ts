import type { AsmLine } from './codegen.js';
import { type AsmInstr, asmText, type Encoded, encode, type SegKind } from './x86.js';

export interface AsmInstrRecord {
  index: number;
  fn: string;
  irId: number | null;
  role: string;
  text: string;
  start: number;
  end: number;
  bytes: number[];
  segments: { kind: SegKind; bytes: number[] }[];
}
export interface Reloc {
  offset: number;
  type: 'R_X86_64_PLT32';
  symbol: string;
  addend: number;
  instrIndex: number;
}
export interface AsmSymbol {
  name: string;
  offset: number;
  size: number;
}
export interface Assembled {
  bytes: number[];
  instrs: AsmInstrRecord[]; // text -> byte range table
  symbols: AsmSymbol[]; // functions defined in this unit
  relocs: Reloc[]; // calls awaiting the linker
  labels: Record<string, number>;
}

/**
 * Assembler: structured instructions -> real bytes.
 * Pass 1 encodes everything (branches get zero rel32 placeholders; all branches are the fixed-size
 * rel32 form so no relaxation is needed and offsets are final immediately).
 * Pass 2 patches intra-unit label branches. Calls to named functions are NOT patched here:
 * they become R_X86_64_PLT32 relocations with addend -4 (the rel32 is measured from the end of
 * the 4-byte field, i.e. P+4, so the linker computes S + A - P with A = -4).
 */
export function assemble(lines: (AsmInstr | AsmLine)[]): Assembled {
  const bytes: number[] = [];
  const instrs: AsmInstrRecord[] = [];
  const labels: Record<string, number> = {};
  const globals: { name: string; offset: number }[] = [];
  const relocs: Reloc[] = [];
  const branchFixups: { at: number; label: string; instr: number }[] = [];

  for (const ins of lines) {
    if (ins.op === 'label') {
      labels[ins.name!] = bytes.length;
      if (ins.global) globals.push({ name: ins.name!, offset: bytes.length });
      continue;
    }
    const enc: Encoded = encode(ins);
    const start = bytes.length;
    bytes.push(...enc.bytes);
    const idx = instrs.length;
    const l = ins as Partial<AsmLine>;
    instrs.push({
      index: idx,
      fn: globals.at(-1)?.name ?? '',
      irId: l.irId ?? null,
      role: l.role ?? 'body',
      text: asmText(ins),
      start,
      end: bytes.length,
      bytes: enc.bytes,
      segments: enc.segments,
    });
    if (enc.fixup?.label !== undefined) branchFixups.push({ at: bytes.length - 4, label: enc.fixup.label, instr: idx });
    if (enc.fixup?.sym !== undefined) {
      relocs.push({
        offset: bytes.length - 4,
        type: 'R_X86_64_PLT32',
        symbol: enc.fixup.sym,
        addend: -4,
        instrIndex: idx,
      });
    }
  }

  for (const f of branchFixups) {
    const target = labels[f.label];
    if (target === undefined) throw new Error(`assembler: undefined label ${f.label}`);
    const rel = target - (f.at + 4); // relative to the end of the instruction
    const patched = [rel & 0xff, (rel >> 8) & 0xff, (rel >> 16) & 0xff, (rel >> 24) & 0xff];
    for (let i = 0; i < 4; i++) bytes[f.at + i] = patched[i];
    const rec = instrs[f.instr];
    rec.bytes = bytes.slice(rec.start, rec.end);
    const last = rec.segments.at(-1)!;
    last.bytes = patched;
  }

  const symbols = globals.map((g, i) => ({
    name: g.name,
    offset: g.offset,
    size: (globals[i + 1]?.offset ?? bytes.length) - g.offset,
  }));
  return { bytes, instrs, symbols, relocs, labels };
}
