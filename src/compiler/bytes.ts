/** Little-endian byte writer used by the ELF writers. */
export class ByteWriter {
  readonly buf: number[] = [];
  get length() {
    return this.buf.length;
  }
  u8(v: number) {
    this.buf.push(v & 0xff);
    return this;
  }
  u16(v: number) {
    return this.u8(v).u8(v >> 8);
  }
  u32(v: number) {
    return this.u16(v & 0xffff).u16(v >>> 16);
  }
  // Numbers up to 2^53 are exact, which covers every address and size here. BigInt handles negatives (addends).
  u64(v: number) {
    const b = BigInt.asUintN(64, BigInt(v));
    return this.u32(Number(b & 0xffffffffn)).u32(Number(b >> 32n));
  }
  bytes(b: ArrayLike<number>) {
    for (let i = 0; i < b.length; i++) this.buf.push(b[i]);
    return this;
  }
  pad(to: number) {
    while (this.buf.length < to) this.buf.push(0);
    return this;
  }
  align(a: number) {
    return this.pad(Math.ceil(this.buf.length / a) * a);
  }
}

/** Little-endian reader over a plain number[] (the serialisable form every stage hands around). */
export class ByteReader {
  readonly view: DataView;
  constructor(public readonly data: number[]) {
    this.view = new DataView(Uint8Array.from(data).buffer);
  }
  u8(o: number) {
    return this.view.getUint8(o);
  }
  u16(o: number) {
    return this.view.getUint16(o, true);
  }
  u32(o: number) {
    return this.view.getUint32(o, true);
  }
  i32(o: number) {
    return this.view.getInt32(o, true);
  }
  u64(o: number) {
    return Number(this.view.getBigUint64(o, true));
  }
  i64(o: number) {
    return Number(this.view.getBigInt64(o, true));
  }
  cstr(o: number) {
    let s = '';
    for (let c = this.data[o]; c; c = this.data[++o]) s += String.fromCharCode(c);
    return s;
  }
}

export const hex = (v: number, w = 0) => `0x${v.toString(16).padStart(w, '0')}`;
