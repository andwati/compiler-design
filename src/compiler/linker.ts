import { ByteWriter } from './bytes.js';
import {
  EHDR_SIZE,
  type ElfHeader,
  type ElfProgramHeader,
  type ElfSection,
  EM_X86_64,
  ET_EXEC,
  PF,
  PHDR_SIZE,
  PT,
  parseElf,
  R_X86_64_PLT32,
  SHF,
  SHT,
  STB,
  STT,
  SYM_SIZE,
  writeEhdr,
} from './elf.js';

export const BASE_ADDR = 0x400000;
const PAGE = 0x1000;
const TEXT_ALIGN = 16;

export interface NamedObject {
  name: string;
  bytes: number[];
}

export type LinkEvent =
  | { type: 'merge'; object: string; section: '.text'; objectOffset: 0; outOffset: number; addr: number; size: number }
  | { type: 'define'; symbol: string; object: string; addr: number }
  | { type: 'resolve'; symbol: string; fromObject: string; definedIn: string; addr: number }
  | {
      type: 'patch';
      object: string;
      symbol: string;
      place: number;
      fileOffset: number;
      S: number;
      A: number;
      P: number;
      value: number;
      before: number[];
      after: number[];
    };

export interface LinkResult {
  bytes: number[];
  header: ElfHeader;
  programHeaders: ElfProgramHeader[];
  sections: ElfSection[];
  symbols: { name: string; addr: number; size: number; object: string }[];
  objects: { name: string; textOffset: number; textAddr: number; size: number }[];
  textAddr: number;
  entry: number;
  events: LinkEvent[];
}

const alignUp = (v: number, a: number) => Math.ceil(v / a) * a;

/**
 * Static linker. Inputs are object *files* (bytes); everything is read back through the ELF parser,
 * so a malformed object fails here exactly as it would for ld.
 *
 * Layout (what GNU ld also produces, minus the extras):
 *   0x400000  ELF header + program headers          PT_LOAD  R
 *   0x401000  .text (objects concatenated in order) PT_LOAD  R+X
 *   (no memory)                                     PT_GNU_STACK RW  (non-executable stack)
 */
export function link(inputs: NamedObject[], entrySymbol = '_start'): LinkResult {
  const events: LinkEvent[] = [];
  const parsed = inputs.map((o) => ({ ...o, elf: parseElf(o.bytes) }));

  const phCount = 3;
  const textFileOff = PAGE; // keep file offset congruent to vaddr modulo page size
  const textAddr = BASE_ADDR + textFileOff;

  // ---- 1. merge .text sections ----
  const text: number[] = [];
  const objects: LinkResult['objects'] = [];
  for (const o of parsed) {
    const sec = o.elf.sections.find((s) => s.name === '.text');
    if (!sec) throw new Error(`link: ${o.name} has no .text`);
    while (text.length % TEXT_ALIGN) text.push(0x90); // pad with NOPs between objects, as ld does for code
    const outOffset = text.length;
    text.push(...o.bytes.slice(sec.offset, sec.offset + sec.size));
    objects.push({ name: o.name, textOffset: outOffset, textAddr: textAddr + outOffset, size: sec.size });
    events.push({
      type: 'merge',
      object: o.name,
      section: '.text',
      objectOffset: 0,
      outOffset,
      addr: textAddr + outOffset,
      size: sec.size,
    });
  }

  // ---- 2. global symbol table ----
  const globals = new Map<string, { addr: number; size: number; object: string }>();
  parsed.forEach((o, i) => {
    for (const s of o.elf.symbols) {
      if (s.bind !== STB.GLOBAL || s.shndx === 0) continue;
      if (globals.has(s.name))
        throw new Error(`link: duplicate symbol '${s.name}' (in ${globals.get(s.name)!.object} and ${o.name})`);
      const addr = objects[i].textAddr + s.value;
      globals.set(s.name, { addr, size: s.size, object: o.name });
      events.push({ type: 'define', symbol: s.name, object: o.name, addr });
    }
  });

  // ---- 3 + 4. resolve undefined symbols and apply relocations ----
  parsed.forEach((o, i) => {
    for (const r of o.elf.relocs) {
      if (r.type !== R_X86_64_PLT32) throw new Error(`link: unsupported relocation type ${r.type}`);
      const sym = o.elf.symbols[r.symIndex];
      let S: number;
      if (sym.type === STT.SECTION) S = objects[i].textAddr;
      else if (sym.shndx !== 0) S = objects[i].textAddr + sym.value;
      else {
        const g = globals.get(sym.name);
        if (!g) throw new Error(`link: undefined reference to '${sym.name}' (in ${o.name})`);
        S = g.addr;
        events.push({ type: 'resolve', symbol: sym.name, fromObject: o.name, definedIn: g.object, addr: g.addr });
      }
      const A = r.addend;
      const place = objects[i].textOffset + r.offset;
      const P = textAddr + place;
      // PLT32 on a statically linked, non-PIE target is a plain 32-bit PC-relative displacement: S + A - P.
      const value = S + A - P;
      if (value < -(2 ** 31) || value >= 2 ** 31) throw new Error(`link: relocation to '${sym.name}' out of range`);
      const before = text.slice(place, place + 4);
      const v = value >>> 0;
      for (let k = 0; k < 4; k++) text[place + k] = (v >>> (8 * k)) & 0xff;
      events.push({
        type: 'patch',
        object: o.name,
        symbol: sym.name,
        place,
        fileOffset: textFileOff + place,
        S,
        A,
        P,
        value,
        before,
        after: text.slice(place, place + 4),
      });
    }
  });

  const entry = globals.get(entrySymbol)?.addr;
  if (entry === undefined) throw new Error(`link: entry symbol '${entrySymbol}' is not defined`);

  // ---- 5. write the executable ----
  const symbols = [...globals].map(([name, g]) => ({ name, ...g }));
  const strBytes: number[] = [0];
  const strOffsets = symbols.map((s) => {
    const o = strBytes.length;
    for (const ch of s.name) strBytes.push(ch.charCodeAt(0));
    strBytes.push(0);
    return o;
  });
  const symw = new ByteWriter();
  symw.pad(SYM_SIZE); // null symbol
  for (const [i, s] of symbols.entries()) {
    symw
      .u32(strOffsets[i])
      .u8((STB.GLOBAL << 4) | STT.FUNC)
      .u8(0)
      .u16(1)
      .u64(s.addr)
      .u64(s.size);
  }
  const shstr: number[] = [0];
  const shName = (n: string) => {
    const o = shstr.length;
    for (const ch of n) shstr.push(ch.charCodeAt(0));
    shstr.push(0);
    return o;
  };
  const nm = ['.text', '.symtab', '.strtab', '.shstrtab'].map(shName);

  const w = new ByteWriter();
  w.pad(textFileOff);
  w.bytes(text);
  const symOff = alignUp(w.length, 8);
  w.pad(symOff);
  w.bytes(symw.buf);
  const strOff = w.length;
  w.bytes(strBytes);
  const shstrOff = w.length;
  w.bytes(shstr);
  const shoff = alignUp(w.length, 8);
  w.pad(shoff);

  const sections: ElfSection[] = [
    { index: 0, name: '', type: 0, flags: 0, addr: 0, offset: 0, size: 0, link: 0, info: 0, addralign: 0, entsize: 0 },
    {
      index: 1,
      name: '.text',
      type: SHT.PROGBITS,
      flags: SHF.ALLOC | SHF.EXEC,
      addr: textAddr,
      offset: textFileOff,
      size: text.length,
      link: 0,
      info: 0,
      addralign: 16,
      entsize: 0,
    },
    {
      index: 2,
      name: '.symtab',
      type: SHT.SYMTAB,
      flags: 0,
      addr: 0,
      offset: symOff,
      size: symw.length,
      link: 3,
      info: 1,
      addralign: 8,
      entsize: SYM_SIZE,
    },
    {
      index: 3,
      name: '.strtab',
      type: SHT.STRTAB,
      flags: 0,
      addr: 0,
      offset: strOff,
      size: strBytes.length,
      link: 0,
      info: 0,
      addralign: 1,
      entsize: 0,
    },
    {
      index: 4,
      name: '.shstrtab',
      type: SHT.STRTAB,
      flags: 0,
      addr: 0,
      offset: shstrOff,
      size: shstr.length,
      link: 0,
      info: 0,
      addralign: 1,
      entsize: 0,
    },
  ];
  for (const s of sections) {
    const n = s.index === 0 ? 0 : nm[s.index - 1];
    w.u32(n)
      .u32(s.type)
      .u64(s.flags)
      .u64(s.addr)
      .u64(s.offset)
      .u64(s.size)
      .u32(s.link)
      .u32(s.info)
      .u64(s.addralign)
      .u64(s.entsize);
  }

  const headersSize = EHDR_SIZE + phCount * PHDR_SIZE;
  const programHeaders: ElfProgramHeader[] = [
    { type: PT.LOAD, flags: PF.R, offset: 0, vaddr: BASE_ADDR, filesz: headersSize, memsz: headersSize, align: PAGE },
    {
      type: PT.LOAD,
      flags: PF.R | PF.X,
      offset: textFileOff,
      vaddr: textAddr,
      filesz: text.length,
      memsz: text.length,
      align: PAGE,
    },
    { type: PT.GNU_STACK, flags: PF.R | PF.W, offset: 0, vaddr: 0, filesz: 0, memsz: 0, align: 16 },
  ];
  const header: ElfHeader = {
    type: ET_EXEC,
    machine: EM_X86_64,
    entry,
    phoff: EHDR_SIZE,
    shoff,
    phnum: phCount,
    shnum: sections.length,
    shstrndx: 4,
  };
  const head = new ByteWriter();
  writeEhdr(head, header);
  for (const p of programHeaders)
    head.u32(p.type).u32(p.flags).u64(p.offset).u64(p.vaddr).u64(p.vaddr).u64(p.filesz).u64(p.memsz).u64(p.align);
  head.buf.forEach((b, i) => {
    w.buf[i] = b;
  });

  return { bytes: w.buf, header, programHeaders, sections, symbols, objects, textAddr, entry, events };
}
