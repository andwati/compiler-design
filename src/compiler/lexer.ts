import { CompileError } from './errors.js';
import type { Comment, Lexed, Token, TokenKind } from './types.js';

const KEYWORDS = new Set(['fn', 'extern', 'let', 'mut', 'while', 'if', 'else', 'return', 'true', 'false']);
const TYPES = new Set(['i32', 'bool']);
const TWO = new Set(['==', '->']);
const OPS = new Set(['+', '-', '*', '<', '>', '==', '=']);
const PUNCT = new Set([';', ':', ',', '(', ')', '{', '}', '->']);

// charMap legend: charMap[i] >= 0 -> index of the token owning char i
//                 CHAR_SPACE      -> whitespace (falls away)
//                 CHAR_COMMENT    -> inside a line comment (falls away)
export const CHAR_SPACE = -1;
export const CHAR_COMMENT = -2;

/**
 * Lexer: character stream -> Token[].
 * Returns the per-character mapping so the scene can show each source
 * character becoming part of a token (or being discarded).
 */
export function lex(src: string): Lexed {
  const tokens: Token[] = [];
  const comments: Comment[] = [];
  const charMap = new Array(src.length).fill(CHAR_SPACE);
  let i = 0,
    line = 1,
    col = 1;

  const adv = () => {
    if (src[i] === '\n') {
      line++;
      col = 1;
    } else col++;
    i++;
  };
  const isAlpha = (c: string) => /[A-Za-z_]/.test(c);
  const isDigit = (c: string) => /[0-9]/.test(c);

  while (i < src.length) {
    const c = src[i];
    if (c === ' ' || c === '\t' || c === '\n' || c === '\r') {
      adv();
      continue;
    }

    const start = i,
      sLine = line,
      sCol = col;
    if (c === '/' && src[i + 1] === '/') {
      while (i < src.length && src[i] !== '\n') {
        charMap[i] = CHAR_COMMENT;
        adv();
      }
      comments.push({ start, end: i, text: src.slice(start, i), line: sLine, col: sCol });
      continue;
    }

    let kind: TokenKind, text: string;
    if (isAlpha(c)) {
      while (i < src.length && (isAlpha(src[i]) || isDigit(src[i]))) adv();
      text = src.slice(start, i);
      kind = KEYWORDS.has(text) ? 'kw' : TYPES.has(text) ? 'type' : 'ident';
    } else if (isDigit(c)) {
      while (i < src.length && isDigit(src[i])) adv();
      text = src.slice(start, i);
      kind = 'int';
    } else {
      const two = src.slice(i, i + 2);
      if (TWO.has(two)) {
        adv();
        adv();
        text = two;
      } else {
        adv();
        text = c;
      }
      if (OPS.has(text)) kind = 'op';
      else if (PUNCT.has(text)) kind = 'punct';
      else throw new CompileError('lexer', `unexpected character '${c}' at ${sLine}:${sCol}`, start, i);
    }
    const idx = tokens.length;
    tokens.push({ kind, text, start, end: i, line: sLine, col: sCol });
    for (let k = start; k < i; k++) charMap[k] = idx;
  }
  tokens.push({ kind: 'eof', text: '', start: src.length, end: src.length, line, col });
  return { source: src, tokens, comments, charMap };
}
