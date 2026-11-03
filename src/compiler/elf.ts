import { ByteReader, ByteWriter } from './bytes.js';

// ---- constants (values straight from the ELF64 / x86-64 psABI) ----
export const ET_REL = 1,
  ET_EXEC = 2,
  EM_X86_64 = 62;
export const SHT = { NULL: 0, PROGBITS: 1, SYMTAB: 2, STRTAB: 3, RELA: 4 } as const;
export const SHF = { WRITE: 1, ALLOC: 2, EXEC: 4, INFO_LINK: 0x40 } as const;
export const STB = { LOCAL: 0, GLOBAL: 1 } as const;
export const STT = { NOTYPE: 0, FUNC: 2, SECTION: 3 } as const;
export const R_X86_64_PLT32 = 4;
export const PT = { LOAD: 1, GNU_STACK: 0x6474e551 } as const;
export const PF = { X: 1, W: 2, R: 4 } as const;
export const EHDR_SIZE = 64,
  PHDR_SIZE = 56,
  SHDR_SIZE = 64,
  SYM_SIZE = 24,
  RELA_SIZE = 24;

export interface ElfHeader {
  type: number;
  machine: number;
  entry: number;
  phoff: number;
  shoff: number;
  phnum: number;
  shnum: number;
  shstrndx: number;
}
export interface ElfSection {
  index: number;
  name: string;
  type: number;
  flags: number;
  addr: number;
  offset: number;
  size: number;
  link: number;
  info: number;
  addralign: number;
  entsize: number;
}
export interface ElfSymbol {
  index: number;
  name: string;
  value: number;
  size: number;
  bind: number;
  type: number;
  shndx: number;
}
export interface ElfReloc {
  offset: number;
  symIndex: number;
  type: number;
  addend: number;
}
export interface ElfProgramHeader {
  type: number;
  flags: number;
  offset: number;
  vaddr: number;
  filesz: number;
  memsz: number;
  align: number;
}
export interface ParsedElf {
  header: ElfHeader;
  sections: ElfSection[];
  symbols: ElfSymbol[];
  relocs: ElfReloc[]; // from .rela.text
  programHeaders: ElfProgramHeader[];
}

export interface ObjectInput {
  text: number[];
  symbols: { name: string; offset: number; size: number }[]; // defined global functions
  undefined: string[]; // referenced but not defined
  relocs: { offset: number; symbol: string; addend: number }[];
}
export interface ObjectLayout {
  bytes: number[];
  header: ElfHeader;
  sections: ElfSection[];
  symbols: ElfSymbol[];
  relocs: (ElfReloc & { symbol: string })[];
  /** byte ranges of each structure in the file, for the scene's hex view */
  regions: { name: string; offset: number; size: number }[];
}

/** Build a string table; returns the bytes and a name -> offset function. */
function strtab() {
  const bytes: number[] = [0];
  const seen = new Map<string, number>();
  return {
    bytes,
    add(s: string) {
      if (s === '') return 0;
      const hit = seen.get(s);
      if (hit !== undefined) return hit;
      const off = bytes.length;
      for (const ch of s) bytes.push(ch.charCodeAt(0));
      bytes.push(0);
      seen.set(s, off);
      return off;
    },
  };
}

/**
 * Write a relocatable ELF64 object (ET_REL):
 *   .text .rela.text .symtab .strtab .shstrtab   (+ section header table)
 * Symbol order matters: all LOCAL symbols precede GLOBAL ones, and .symtab's sh_info is the index of
 * the first GLOBAL symbol.
 */
export function writeObject(input: ObjectInput): ObjectLayout {
  const strs = strtab();
  const shstr = strtab();

  // symbols: [null, section(.text) LOCAL, defined GLOBAL FUNC..., undefined GLOBAL NOTYPE...]
  const syms: (ElfSymbol & { symbol?: string })[] = [
    { index: 0, name: '', value: 0, size: 0, bind: 0, type: 0, shndx: 0 },
    { index: 1, name: '', value: 0, size: 0, bind: STB.LOCAL, type: STT.SECTION, shndx: 1 },
  ];
  for (const s of input.symbols)
    syms.push({
      index: syms.length,
      name: s.name,
      value: s.offset,
      size: s.size,
      bind: STB.GLOBAL,
      type: STT.FUNC,
      shndx: 1,
    });
  for (const u of input.undefined)
    syms.push({ index: syms.length, name: u, value: 0, size: 0, bind: STB.GLOBAL, type: STT.NOTYPE, shndx: 0 });
  const firstGlobal = 2;
  const symIndex = new Map(syms.map((s) => [s.name, s.index]));

  const w = new ByteWriter();
  w.pad(EHDR_SIZE); // header is written last, once offsets are known

  const regions: ObjectLayout['regions'] = [{ name: 'ELF header', offset: 0, size: EHDR_SIZE }];
  const sections: ElfSection[] = [];
  const addSection = (
    name: string,
    type: number,
    flags: number,
    data: number[],
    opts: { align?: number; link?: number; info?: number; entsize?: number },
  ) => {
    w.align(opts.align ?? 1);
    const offset = w.length;
    w.bytes(data);
    sections.push({
      index: sections.length + 1,
      name,
      type,
      flags,
      addr: 0,
      offset,
      size: data.length,
      link: opts.link ?? 0,
      info: opts.info ?? 0,
      addralign: opts.align ?? 1,
      entsize: opts.entsize ?? 0,
    });
    regions.push({ name, offset, size: data.length });
  };

  addSection('.text', SHT.PROGBITS, SHF.ALLOC | SHF.EXEC, input.text, { align: 16 });

  const rela = new ByteWriter();
  const relocs: ObjectLayout['relocs'] = [];
  for (const r of input.relocs) {
    const si = symIndex.get(r.symbol);
    if (si === undefined) throw new Error(`elf: relocation against unknown symbol ${r.symbol}`);
    // r_info = (symbol index << 32) | type
    rela
      .u64(r.offset)
      .u64(si * 2 ** 32 + R_X86_64_PLT32)
      .u64(r.addend);
    relocs.push({ offset: r.offset, symIndex: si, type: R_X86_64_PLT32, addend: r.addend, symbol: r.symbol });
  }
  // sections are numbered: 1 .text, 2 .rela.text, 3 .symtab, 4 .strtab, 5 .shstrtab
  addSection('.rela.text', SHT.RELA, SHF.INFO_LINK, rela.buf, { align: 8, link: 3, info: 1, entsize: RELA_SIZE });

  const symw = new ByteWriter();
  for (const s of syms) {
    symw
      .u32(strs.add(s.name))
      .u8((s.bind << 4) | s.type)
      .u8(0)
      .u16(s.shndx)
      .u64(s.value)
      .u64(s.size);
  }
  addSection('.symtab', SHT.SYMTAB, 0, symw.buf, { align: 8, link: 4, info: firstGlobal, entsize: SYM_SIZE });
  addSection('.strtab', SHT.STRTAB, 0, strs.bytes, {});

  // section names must be registered before the table is serialised
  const names = ['.text', '.rela.text', '.symtab', '.strtab', '.shstrtab'];
  const nameOff = names.map((n) => shstr.add(n));
  addSection('.shstrtab', SHT.STRTAB, 0, shstr.bytes, {});

  w.align(8);
  const shoff = w.length;
  w.pad(w.length + SHDR_SIZE); // index 0: SHT_NULL
  for (const s of sections) {
    w.u32(nameOff[s.index - 1])
      .u32(s.type)
      .u64(s.flags)
      .u64(0)
      .u64(s.offset)
      .u64(s.size)
      .u32(s.link)
      .u32(s.info)
      .u64(s.addralign)
      .u64(s.entsize);
  }
  regions.push({ name: 'section headers', offset: shoff, size: SHDR_SIZE * (sections.length + 1) });

  const header: ElfHeader = {
    type: ET_REL,
    machine: EM_X86_64,
    entry: 0,
    phoff: 0,
    shoff,
    phnum: 0,
    shnum: sections.length + 1,
    shstrndx: sections.length,
  };
  const hdr = new ByteWriter();
  writeEhdr(hdr, header);
  for (let i = 0; i < EHDR_SIZE; i++) w.buf[i] = hdr.buf[i];

  return { bytes: w.buf, header, sections, symbols: syms, relocs, regions };
}

export function writeEhdr(w: ByteWriter, h: ElfHeader) {
  // e_ident: magic, ELFCLASS64, little-endian, EV_CURRENT, System V ABI, padding
  w.bytes([0x7f, 0x45, 0x4c, 0x46, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  w.u16(h.type).u16(h.machine).u32(1).u64(h.entry).u64(h.phoff).u64(h.shoff).u32(0);
  w.u16(EHDR_SIZE)
    .u16(h.phnum ? PHDR_SIZE : 0)
    .u16(h.phnum)
    .u16(SHDR_SIZE)
    .u16(h.shnum)
    .u16(h.shstrndx);
}

/** Parse an ELF64 file (objects and executables) back into structured data. */
export function parseElf(data: number[]): ParsedElf {
  const r = new ByteReader(data);
  if (r.u32(0) !== 0x464c457f) throw new Error('elf: bad magic');
  if (r.u8(4) !== 2 || r.u8(5) !== 1) throw new Error('elf: not ELF64 little-endian');
  const header: ElfHeader = {
    type: r.u16(16),
    machine: r.u16(18),
    entry: r.u64(24),
    phoff: r.u64(32),
    shoff: r.u64(40),
    phnum: r.u16(56),
    shnum: r.u16(60),
    shstrndx: r.u16(62),
  };
  const programHeaders: ElfProgramHeader[] = [];
  for (let i = 0; i < header.phnum; i++) {
    const o = header.phoff + i * PHDR_SIZE;
    programHeaders.push({
      type: r.u32(o),
      flags: r.u32(o + 4),
      offset: r.u64(o + 8),
      vaddr: r.u64(o + 16),
      filesz: r.u64(o + 32),
      memsz: r.u64(o + 40),
      align: r.u64(o + 48),
    });
  }
  const shdrs = [];
  for (let i = 0; i < header.shnum; i++) {
    const o = header.shoff + i * SHDR_SIZE;
    shdrs.push({
      nameOff: r.u32(o),
      index: i,
      type: r.u32(o + 4),
      flags: r.u64(o + 8),
      addr: r.u64(o + 16),
      offset: r.u64(o + 24),
      size: r.u64(o + 32),
      link: r.u32(o + 40),
      info: r.u32(o + 44),
      addralign: r.u64(o + 48),
      entsize: r.u64(o + 56),
    });
  }
  const shstrOff = header.shnum ? shdrs[header.shstrndx].offset : 0;
  const sections: ElfSection[] = shdrs.map(({ nameOff, ...s }) => ({ ...s, name: r.cstr(shstrOff + nameOff) }));

  const symbols: ElfSymbol[] = [];
  const symtab = sections.find((s) => s.type === SHT.SYMTAB);
  if (symtab) {
    const strOff = sections[symtab.link].offset;
    for (let i = 0; i < symtab.size / SYM_SIZE; i++) {
      const o = symtab.offset + i * SYM_SIZE;
      const info = r.u8(o + 4);
      symbols.push({
        index: i,
        name: r.cstr(strOff + r.u32(o)),
        bind: info >> 4,
        type: info & 15,
        shndx: r.u16(o + 6),
        value: r.u64(o + 8),
        size: r.u64(o + 16),
      });
    }
  }
  const relocs: ElfReloc[] = [];
  const relaSec = sections.find((s) => s.type === SHT.RELA);
  if (relaSec) {
    for (let i = 0; i < relaSec.size / RELA_SIZE; i++) {
      const o = relaSec.offset + i * RELA_SIZE;
      const info = r.u64(o + 8);
      relocs.push({
        offset: r.u64(o),
        symIndex: Math.floor(info / 2 ** 32),
        type: info % 2 ** 32,
        addend: r.i64(o + 16),
      });
    }
  }
  return { header, sections, symbols, relocs, programHeaders };
}
