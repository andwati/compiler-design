import { describe, expect, it } from 'vitest';
import { assemble } from '../src/compiler/assembler.ts';
import { type ParsedElf, parseElf } from '../src/compiler/elf.ts';
import { buildCrt0, buildProgram, compileUnit, MAIN_RS, MATH_RS } from '../src/compiler/index.ts';
import { BASE_ADDR, link } from '../src/compiler/linker.ts';
import type { AsmInstr } from '../src/compiler/x86.ts';
import { encode, imm, mem, reg } from '../src/compiler/x86.ts';
import { exitCodeOf, hasTool, run2, tmpFile } from './helpers.ts';

const hexOf = (i: AsmInstr) =>
  encode(i)
    .bytes.map((b) => b.toString(16).padStart(2, '0'))
    .join(' ');

describe('x86-64 encoder: known-good bytes (from the Intel SDM / objdump)', () => {
  const cases: [string, AsmInstr, string][] = [
    ['push rbp', { op: 'push', size: 64, dst: reg('rbp') }, '55'],
    ['pop rbp', { op: 'pop', size: 64, dst: reg('rbp') }, '5d'],
    ['push r12 (REX.B)', { op: 'push', size: 64, dst: reg('r12') }, '41 54'],
    ['pop r12', { op: 'pop', size: 64, dst: reg('r12') }, '41 5c'],
    ['ret', { op: 'ret' }, 'c3'],
    ['syscall', { op: 'syscall' }, '0f 05'],
    ['mov rbp, rsp (REX.W)', { op: 'mov', size: 64, dst: reg('rbp'), src: reg('rsp') }, '48 8b ec'],
    ['mov eax, ebx', { op: 'mov', size: 32, dst: reg('rax'), src: reg('rbx') }, '8b c3'],
    ['mov r12d, edi (REX.R)', { op: 'mov', size: 32, dst: reg('r12'), src: reg('rdi') }, '44 8b e7'],
    ['mov edi, r12d (REX.B)', { op: 'mov', size: 32, dst: reg('rdi'), src: reg('r12') }, '41 8b fc'],
    ['mov eax, 60', { op: 'mov', size: 32, dst: reg('rax'), src: imm(60) }, 'b8 3c 00 00 00'],
    ['mov r10d, 1 (REX.B + B8+rd)', { op: 'mov', size: 32, dst: reg('r10'), src: imm(1) }, '41 ba 01 00 00 00'],
    ['mov [rbp-4], eax', { op: 'mov', size: 32, dst: mem(-4), src: reg('rax') }, '89 45 fc'],
    ['mov eax, [rbp-8]', { op: 'mov', size: 32, dst: reg('rax'), src: mem(-8) }, '8b 45 f8'],
    ['mov dword [rbp-4], 5', { op: 'mov', size: 32, dst: mem(-4), src: imm(5) }, 'c7 45 fc 05 00 00 00'],
    ['mov eax, [rbp-200] (disp32)', { op: 'mov', size: 32, dst: reg('rax'), src: mem(-200) }, '8b 85 38 ff ff ff'],
    ['add eax, ebx', { op: 'add', size: 32, dst: reg('rax'), src: reg('rbx') }, '03 c3'],
    ['add ebx, 5 (imm8)', { op: 'add', size: 32, dst: reg('rbx'), src: imm(5) }, '83 c3 05'],
    ['add eax, 1000 (imm32)', { op: 'add', size: 32, dst: reg('rax'), src: imm(1000) }, '81 c0 e8 03 00 00'],
    ['sub eax, [rbp-4]', { op: 'sub', size: 32, dst: reg('rax'), src: mem(-4) }, '2b 45 fc'],
    ['sub rsp, 16 (REX.W)', { op: 'sub', size: 64, dst: reg('rsp'), src: imm(16) }, '48 83 ec 10'],
    ['add rsp, 16', { op: 'add', size: 64, dst: reg('rsp'), src: imm(16) }, '48 83 c4 10'],
    ['cmp ebx, 5', { op: 'cmp', size: 32, dst: reg('rbx'), src: imm(5) }, '83 fb 05'],
    ['cmp r12d, 5', { op: 'cmp', size: 32, dst: reg('r12'), src: imm(5) }, '41 83 fc 05'],
    ['cmp dword [rbp-4], 0', { op: 'cmp', size: 32, dst: mem(-4), src: imm(0) }, '83 7d fc 00'],
    ['cmp eax, ebx', { op: 'cmp', size: 32, dst: reg('rax'), src: reg('rbx') }, '3b c3'],
    ['imul eax, ebx', { op: 'imul', dst: reg('rax'), src: reg('rbx') }, '0f af c3'],
    ['imul r10d, r10d', { op: 'imul', dst: reg('r10'), src: reg('r10') }, '45 0f af d2'],
    ['imul eax, ebx, 3', { op: 'imul3', dst: reg('rax'), src: reg('rbx'), imm: 3 }, '6b c3 03'],
    ['imul eax, ebx, 1000', { op: 'imul3', dst: reg('rax'), src: reg('rbx'), imm: 1000 }, '69 c3 e8 03 00 00'],
    ['setl al', { op: 'setcc', cond: 'l', dst: reg('rax') }, '0f 9c c0'],
    ['setg al', { op: 'setcc', cond: 'g', dst: reg('rax') }, '0f 9f c0'],
    ['sete al', { op: 'setcc', cond: 'e', dst: reg('rax') }, '0f 94 c0'],
    ['movzx eax, al', { op: 'movzx', dst: reg('rax'), src: reg('rax') }, '0f b6 c0'],
    ['jmp rel32 placeholder', { op: 'jmp', target: 'x' }, 'e9 00 00 00 00'],
    ['jl rel32 placeholder', { op: 'jcc', cond: 'l', target: 'x' }, '0f 8c 00 00 00 00'],
    ['jge rel32 placeholder', { op: 'jcc', cond: 'ge', target: 'x' }, '0f 8d 00 00 00 00'],
    ['call rel32 placeholder', { op: 'call', sym: 'f' }, 'e8 00 00 00 00'],
  ];
  for (const [name, ins, expected] of cases) it(name, () => expect(hexOf(ins)).toBe(expected));

  it('splits bytes into role segments (rex / opcode / modrm / imm)', () => {
    const e = encode({ op: 'cmp', size: 32, dst: reg('r12'), src: imm(5) });
    expect(e.segments.map((s) => s.kind)).toEqual(['rex', 'opcode', 'modrm', 'imm']);
  });

  it('patches intra-unit jumps with the right rel32', () => {
    const a = assemble([
      { op: 'label', name: 'top' },
      { op: 'ret' }, // 1 byte at 0
      { op: 'jmp', target: 'top' }, // e9 at 1..5, next ip = 6
    ]);
    // target 0, next ip 6 -> rel = -6 = fa ff ff ff
    expect(a.bytes).toEqual([0xc3, 0xe9, 0xfa, 0xff, 0xff, 0xff]);
  });

  it('call to a named function leaves a PLT32 relocation with addend -4 and a zero placeholder', () => {
    const a = assemble([{ op: 'label', name: 'f', global: true }, { op: 'call', sym: 'g' }, { op: 'ret' }]);
    expect(a.bytes).toEqual([0xe8, 0, 0, 0, 0, 0xc3]);
    expect(a.relocs).toEqual([{ offset: 1, type: 'R_X86_64_PLT32', symbol: 'g', addend: -4, instrIndex: 0 }]);
  });
});

describe('assembler tables', () => {
  const main = compileUnit('main.rs', MAIN_RS);
  it('instruction table tiles .text exactly (no gaps, no overlaps)', () => {
    let at = 0;
    for (const i of main.assembled.instrs) {
      expect(i.start).toBe(at);
      at = i.end;
      expect(i.bytes).toHaveLength(i.end - i.start);
    }
    expect(at).toBe(main.assembled.bytes.length);
    for (const i of main.assembled.instrs) expect(main.assembled.bytes.slice(i.start, i.end)).toEqual(i.bytes);
  });
  it('every instruction row records the IR instruction it came from (except prologue/epilogue glue)', () => {
    const body = main.assembled.instrs.filter((i) => i.role === 'body');
    expect(body.every((i) => i.irId !== null || /^(mov|add|sub|cmp)/.test(i.text) || i.text.startsWith('j'))).toBe(
      true,
    );
  });
});

describe.runIf(hasTool('objdump'))('cross-check against objdump', () => {
  // Disassemble our bytes with the system disassembler and require identical instruction
  // boundaries and mnemonics: any wrong prefix/modrm/length would shift the decode.
  for (const [name, bytes, instrs] of [
    ['main.o', compileUnit('main.rs', MAIN_RS).object.bytes, compileUnit('main.rs', MAIN_RS).assembled.instrs],
    ['math.o', compileUnit('math.rs', MATH_RS).object.bytes, compileUnit('math.rs', MATH_RS).assembled.instrs],
    ['crt0.o', buildCrt0().object.bytes, buildCrt0().assembled.instrs],
  ] as const) {
    it(`${name}: same boundaries and mnemonics`, () => {
      const out = run2('objdump', ['-d', '-M', 'intel', '--no-show-raw-insn', tmpFile(name, bytes)]);
      const lines = [...out.matchAll(/^\s+([0-9a-f]+):\s+(\S+)/gm)].map((m) => ({
        off: Number.parseInt(m[1], 16),
        mn: m[2],
      }));
      expect(lines.map((l) => l.off)).toEqual(instrs.map((i) => i.start));
      expect(lines.map((l) => l.mn)).toEqual(instrs.map((i) => i.text.split(' ')[0]));
    });
  }
});

describe('ELF relocatable objects', () => {
  const main = compileUnit('main.rs', MAIN_RS);
  const elf: ParsedElf = parseElf(main.object.bytes);

  it('has the required sections', () => {
    expect(elf.sections.map((s) => s.name)).toEqual(['', '.text', '.rela.text', '.symtab', '.strtab', '.shstrtab']);
    expect(elf.header.type).toBe(1); // ET_REL
    expect(elf.header.machine).toBe(62);
  });
  it('emits R_X86_64_PLT32 against the undefined symbol `square`', () => {
    expect(elf.relocs).toHaveLength(1);
    const r = elf.relocs[0];
    expect(r.type).toBe(4);
    expect(r.addend).toBe(-4);
    const sym = elf.symbols[r.symIndex];
    expect(sym.name).toBe('square');
    expect(sym.shndx).toBe(0); // SHN_UNDEF
    // r_offset points at the rel32 field, one byte after the E8 opcode
    expect(main.object.bytes[elf.sections[1].offset + r.offset - 1]).toBe(0xe8);
  });
  it('defines main as a global function and keeps locals before globals', () => {
    const m = elf.symbols.find((s) => s.name === 'main')!;
    expect(m.bind).toBe(1);
    expect(m.type).toBe(2);
    expect(m.shndx).toBe(1);
    expect(m.size).toBe(main.assembled.bytes.length);
    const symtab = elf.sections.find((s) => s.name === '.symtab')!;
    expect(symtab.info).toBe(2); // first global index
  });
  it.runIf(hasTool('readelf'))('is accepted by readelf', () => {
    const f = tmpFile('main.o', main.object.bytes);
    const rel = run2('readelf', ['-r', f]);
    expect(rel).toMatch(/R_X86_64_PLT32\s+0+\s+square - 4/);
    const syms = run2('readelf', ['-s', f]);
    expect(syms).toMatch(/FUNC\s+GLOBAL DEFAULT\s+1 main/);
    expect(syms).toMatch(/NOTYPE\s+GLOBAL DEFAULT\s+UND square/);
  });
});

describe('linker', () => {
  const p = buildProgram();
  const sym = (n: string) => p.linked.symbols.find((s) => s.name === n)!;
  const exe = parseElf(p.linked.bytes);

  it('merges objects in command-line order from 0x401000 with 16-byte alignment', () => {
    expect(p.linked.objects.map((o) => o.name)).toEqual(['main.o', 'math.o', 'crt0.o']);
    expect(p.linked.textAddr).toBe(0x401000);
    let off = 0;
    for (const [i, o] of p.linked.objects.entries()) {
      expect(o.textOffset).toBe(off);
      expect(o.textOffset % 16).toBe(0);
      off = Math.ceil((o.textOffset + o.size) / 16) * 16;
      void i;
    }
  });

  it('resolves `square` and `main` and puts _start at the entry point', () => {
    expect(sym('main').addr).toBe(0x401000);
    expect(sym('square').addr).toBe(0x401000 + p.linked.objects[1].textOffset);
    expect(p.linked.entry).toBe(sym('_start').addr);
    expect(exe.header.entry).toBe(sym('_start').addr);
    expect(exe.header.type).toBe(2); // ET_EXEC
  });

  it('relocation value equals an independent hand calculation', () => {
    // Hand calculation, using only the layout facts above and the compiler's own instruction table:
    //   P = address of the rel32 field = address of main.o's .text + offset of the call's displacement
    //   S = address of square, A = -4  =>  disp = S + A - P
    const mainUnit = p.main;
    const call = mainUnit.assembled.instrs.find((i) => i.text === 'call square')!;
    const P = 0x401000 + call.start + 1;
    const S = 0x401000 + p.linked.objects[1].textOffset;
    const expected = S - 4 - P;
    const patch = p.linked.events.find((e) => e.type === 'patch' && e.symbol === 'square')!;
    expect(patch.type === 'patch' && patch.value).toBe(expected);
    // and the bytes in the executable really are that little-endian int32
    const fileOff = 0x1000 + call.start + 1;
    const got = new DataView(Uint8Array.from(p.linked.bytes.slice(fileOff, fileOff + 4)).buffer).getInt32(0, true);
    expect(got).toBe(expected);
    // target of the call = address after the call instruction + displacement = square
    const next = 0x401000 + call.end;
    expect(next + got).toBe(sym('square').addr);
    expect(patch.type === 'patch' && patch.before).toEqual([0, 0, 0, 0]);
  });

  it('crt0 call to main is patched too', () => {
    const patch = p.linked.events.find((e) => e.type === 'patch' && e.symbol === 'main')!;
    const crt = p.linked.objects[2];
    const P = crt.textAddr + 1; // crt0: call is the first instruction
    expect(patch.type === 'patch' && patch.P).toBe(P);
    expect(patch.type === 'patch' && patch.value).toBe(sym('main').addr - 4 - P);
  });

  it('writes sane program headers', () => {
    const [hdrs, text, stack] = exe.programHeaders;
    expect(hdrs).toMatchObject({ type: 1, flags: 4, vaddr: BASE_ADDR, offset: 0 });
    expect(text).toMatchObject({
      type: 1,
      flags: 5,
      vaddr: 0x401000,
      offset: 0x1000,
      filesz: p.linked.textAddr ? text.memsz : 0,
    });
    expect(stack).toMatchObject({ type: 0x6474e551, flags: 6 });
    expect((text.vaddr - text.offset) % text.align).toBe(0);
  });

  it('fails loudly on undefined symbols and duplicates', () => {
    const main = compileUnit('main.rs', MAIN_RS).object.bytes;
    const crt0 = buildCrt0().object.bytes;
    expect(() =>
      link([
        { name: 'main.o', bytes: main },
        { name: 'crt0.o', bytes: crt0 },
      ]),
    ).toThrow(/undefined reference to 'square'/);
    const math = compileUnit('math.rs', MATH_RS).object.bytes;
    expect(() =>
      link([
        { name: 'a.o', bytes: math },
        { name: 'b.o', bytes: math },
      ]),
    ).toThrow(/duplicate symbol 'square'/);
  });

  it.runIf(hasTool('readelf') && hasTool('objdump'))('produces a file readelf and objdump accept', () => {
    const f = tmpFile('a.out', p.linked.bytes);
    const h = run2('readelf', ['-lh', f]);
    expect(h).toMatch(/Type:\s+EXEC/);
    expect(h).toMatch(new RegExp(`Entry point address:\\s+0x${sym('_start').addr.toString(16)}`));
    expect(h).toMatch(/LOAD\s+0x0+1000 0x0+401000/);
    const d = run2('objdump', ['-d', '-M', 'intel', '--no-show-raw-insn', f]);
    expect(d).toMatch(/call\s+401[0-9a-f]+ <square>/);
    expect(d).toMatch(/call\s+401[0-9a-f]+ <main>/);
  });
});

describe('loader + executor', () => {
  const p = buildProgram();
  it('maps segments and a stack near the top of the address space', () => {
    const maps = p.loaded.events.filter((e) => e.type === 'map');
    expect(maps.map((m) => m.type === 'map' && m.vaddr)).toEqual([0x400000, 0x401000]);
    const stack = p.loaded.events.find((e) => e.type === 'stack')!;
    expect(stack.type === 'stack' && stack.top).toBe(0x7ffffffff000);
  });
  it('starts at _start with a sane rsp, and exits with 35', () => {
    expect(p.execution.steps[0].rip).toBe(p.linked.entry);
    expect(p.execution.exitCode).toBe(35);
    expect(p.execution.steps.at(-1)!.kind).toBe('syscall');
  });
  it('shows the call, the return and a balanced stack', () => {
    const calls = p.execution.steps.filter((s) => s.kind === 'call');
    const rets = p.execution.steps.filter((s) => s.kind === 'ret');
    // _start->main, then 4 x main->square; and returns: 4 x square, 1 x main
    expect(calls).toHaveLength(5);
    expect(rets).toHaveLength(5);
    // _start's call pushed a return address (rsp-8); main's final ret pops it, so rsp is back to its initial value
    const initialRsp = 0x7ffffffff000 - 0x20;
    const afterMainRet = p.execution.steps[p.execution.steps.indexOf(rets.at(-1)!)];
    expect(afterMainRet.regs[4]).toBe(initialRsp);
    expect(p.execution.steps[0].regs[4]).toBe(initialRsp - 8); // right after _start's call
  });
  it('executes the sum 1+4+9+16 in eax before adding 5', () => {
    const retMain = p.execution.steps.filter((s) => s.kind === 'ret').at(-1)!;
    expect(retMain.regs[0]).toBe(35);
  });
  it('interpreter writes are visible as stack stores (push of return address)', () => {
    const call = p.execution.steps.find((s) => s.kind === 'call')!;
    expect(call.writes[0].size).toBe(8);
    expect(call.writes[0].value).toBe(call.rip + call.len);
  });
  it('faults on writes to read-only text', () => {
    expect(() => p.loaded.mem.write32(0x401000, 0)).toThrow(/protection fault/);
  });
});

describe('end to end', () => {
  it('compiles, links and executes the sample to exit code 35', () => {
    expect(buildProgram().execution.exitCode).toBe(35);
  });

  it('is deterministic: identical bytes on every build', () => {
    expect(buildProgram().linked.bytes).toEqual(buildProgram().linked.bytes);
  });

  it('stage outputs are JSON-serialisable', () => {
    const p = buildProgram();
    for (const part of [
      p.main.lexed,
      p.main.parsed,
      p.main.sema,
      p.main.optimized,
      p.main.code,
      p.main.assembled,
      p.main.object,
      p.linked,
      p.execution,
    ]) {
      expect(() => JSON.stringify(part)).not.toThrow();
    }
  });

  it('allocates the sample without spilling, using callee-saved registers across the call', () => {
    const f = compileUnit('main.rs', MAIN_RS).code.functions[0];
    expect(f.spills).toEqual([]);
    const across = f.intervals
      .filter((i) => i.crossesCall)
      .map((i) => i.vreg)
      .sort();
    expect(across).toEqual(['i', 'sum']);
    for (const i of f.intervals.filter((x) => x.crossesCall)) expect(['rbx', 'r12', 'r13']).toContain(i.loc.reg);
  });

  it('SPILLS under register pressure (one callee-saved register) and still exits 35', () => {
    const regs = { calleeSaved: ['rbx'] as const, callerSaved: ['r10'] as const };
    const u = compileUnit('main.rs', MAIN_RS, {
      regs: { calleeSaved: [...regs.calleeSaved], callerSaved: [...regs.callerSaved] },
    });
    const f = u.code.functions[0];
    expect(f.spills.length).toBe(1);
    expect(f.allocEvents.some((e) => e.type === 'spill')).toBe(true);
    expect(f.asm.some((l) => l.text.includes('[rbp-'))).toBe(true);
    const code = buildProgram(
      { math: MATH_RS, main: MAIN_RS },
      { regs: { calleeSaved: ['rbx'], callerSaved: ['r10'] } },
    );
    expect(code.execution.exitCode).toBe(35);
  });

  it('still exits 35 with ZERO callee-saved registers (everything live across the call lives on the stack)', () => {
    const code = buildProgram({ math: MATH_RS, main: MAIN_RS }, { regs: { calleeSaved: [], callerSaved: ['r10'] } });
    expect(code.execution.exitCode).toBe(35);
    expect(code.main.code.functions[0].spills.sort()).toEqual(['i', 'sum']);
  });

  const progs: [string, string, number][] = [
    ['if/else', 'fn main() -> i32 { let x: i32 = 3; let mut r: i32 = 0; if x > 2 { r = 10; } else { r = 20; } r }', 10],
    ['if without else', 'fn main() -> i32 { let mut r: i32 = 5; if r == 4 { r = 9; } r }', 5],
    [
      'early return',
      'fn f(a: i32) -> i32 { if a < 0 { return 1; } return 2; } fn main() -> i32 { f(3) + f(0 - 1) * 10 }',
      12,
    ],
    [
      'bool values via setcc',
      'fn lt(a: i32, b: i32) -> bool { a < b } fn main() -> i32 { let c: bool = lt(2, 3); let mut r: i32 = 1; if c { r = 42; } r }',
      42,
    ],
    [
      'recursion',
      'fn fact(n: i32) -> i32 { if n < 2 { return 1; } n * fact(n - 1) } fn main() -> i32 { fact(5) }',
      120,
    ],
    ['subtraction and operand order', 'fn sub(a: i32, b: i32) -> i32 { a - b } fn main() -> i32 { sub(50, 8) }', 42],
    [
      'six arguments',
      'fn f(a: i32, b: i32, c: i32, d: i32, e: i32, g: i32) -> i32 { a + b * 2 + c * 3 + d * 4 + e * 5 + g * 6 } fn main() -> i32 { f(1, 1, 1, 1, 1, 1) }',
      21,
    ],
    ['multiplication by constant on the left', 'fn f(a: i32) -> i32 { 3 * a } fn main() -> i32 { f(14) }', 42],
    [
      'nested loops',
      'fn main() -> i32 { let mut t: i32 = 0; let mut i: i32 = 0; while i < 4 { let mut j: i32 = 0; while j < 3 { t = t + 1; j = j + 1; } i = i + 1; } t }',
      12,
    ],
    ['negative results wrap into the exit status', 'fn main() -> i32 { 0 - 1 }', 255],
    ['equality', 'fn main() -> i32 { let a: i32 = 4; let mut r: i32 = 0; if a == 4 { r = 1; } r }', 1],
    ['shadowing', 'fn main() -> i32 { let x: i32 = 1; let x: i32 = x + 41; x }', 42],
  ];
  for (const [name, src, want] of progs) {
    it(`program: ${name}`, () => expect(exitCodeOf(src)).toBe(want));
    it(`program under max register pressure: ${name}`, () =>
      expect(exitCodeOf(src, { calleeSaved: [], callerSaved: ['r10'] })).toBe(want));
  }
});
