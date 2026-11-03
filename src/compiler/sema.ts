import type { AstNode, Parsed, SemaError, SemaEvent, SemaResult, Sym, Ty } from './types.js';

interface Scope {
  id: number;
  owner: number;
  label: string;
  names: Map<string, Sym>;
}

/**
 * Semantic analysis: scope stack, name resolution, type checking, mutability check.
 * Works on a deep clone of the AST and annotates it (node.ty, node.symbol), so the
 * parser's output stays untouched (stages are pure functions).
 *
 * events: the symbol-table story in order -
 *   push-scope / define / resolve / pop-scope  (scope chain)
 *   type                                       (a type flowing up to a node)
 */
export function analyze(parsed: Parsed): SemaResult {
  const root = structuredClone(parsed.root);
  const events: SemaEvent[] = [];
  const errors: SemaError[] = [];
  const symbols: Sym[] = [];
  const scopes: Scope[] = [];
  let scopeSeq = 0;

  const error = (node: AstNode, message: string, code: string) =>
    errors.push({ message, code, nodeId: node.id, start: node.span.start, end: node.span.end });

  const pushScope = (owner: AstNode, label: string) => {
    const s: Scope = { id: scopeSeq++, owner: owner.id, label, names: new Map() };
    scopes.push(s);
    events.push({ type: 'push-scope', scopeId: s.id, depth: scopes.length - 1, label, nodeId: owner.id });
  };
  const popScope = () => {
    const s = scopes.pop()!;
    events.push({
      type: 'pop-scope',
      scopeId: s.id,
      depth: scopes.length,
      symbols: [...s.names.values()].map((x) => x.id),
    });
  };
  const define = (node: AstNode, name: string, kind: Sym['kind'], ty: Ty, mut: boolean) => {
    const sym: Sym = {
      id: symbols.length,
      name,
      kind,
      ty,
      mut,
      nodeId: node.id,
      scopeId: scopes.at(-1)!.id,
      depth: scopes.length - 1,
    };
    symbols.push(sym);
    scopes.at(-1)!.names.set(name, sym);
    node.symbol = sym.id;
    events.push({
      type: 'define',
      symbol: sym.id,
      name,
      kind,
      ty,
      mut,
      nodeId: node.id,
      scopeId: sym.scopeId,
      depth: sym.depth,
    });
    return sym;
  };
  const lookup = (name: string): Sym | null => {
    for (let d = scopes.length - 1; d >= 0; d--) {
      const s = scopes[d].names.get(name);
      if (s) return s;
    }
    return null;
  };
  const resolve = (node: AstNode, name: string): Sym | null => {
    const sym = lookup(name);
    if (!sym) {
      error(node, `cannot find value \`${name}\` in this scope`, 'E0425');
      return null;
    }
    node.symbol = sym.id;
    events.push({
      type: 'resolve',
      nodeId: node.id,
      name,
      symbol: sym.id,
      fromDepth: scopes.length - 1,
      toDepth: sym.depth,
    });
    return sym;
  };
  const setType = (node: AstNode, ty: Ty): Ty => {
    node.ty = ty;
    events.push({ type: 'type', nodeId: node.id, ty });
    return ty;
  };

  let currentFn!: AstNode;

  function expr(n: AstNode): Ty {
    switch (n.kind) {
      case 'Int':
        return setType(n, 'i32');
      case 'Bool':
        return setType(n, 'bool');
      case 'Ident': {
        const sym = resolve(n, n.name);
        return setType(n, sym ? sym.ty : 'error');
      }
      case 'Binary': {
        const l = expr(n.left),
          r = expr(n.right);
        const arith = n.op === '+' || n.op === '-' || n.op === '*';
        const cmp = n.op === '<' || n.op === '>';
        if (l === 'error' || r === 'error') return setType(n, 'error');
        if (arith || cmp) {
          if (l !== 'i32' || r !== 'i32') {
            error(n, `mismatched types: \`${n.op}\` needs i32 operands, found ${l} and ${r}`, 'E0308');
            return setType(n, 'error');
          }
          return setType(n, arith ? 'i32' : 'bool');
        }
        if (l !== r) {
          error(n, `mismatched types: cannot compare ${l} with ${r}`, 'E0308');
          return setType(n, 'error');
        }
        return setType(n, 'bool');
      }
      case 'Call': {
        const sym = resolve(n.callee, n.callee.name);
        const argTys = n.args.map(expr);
        if (!sym) return setType(n, 'error');
        setType(n.callee, 'fn');
        if (sym.kind !== 'fn') {
          error(n, `\`${n.callee.name}\` is not a function`, 'E0618');
          return setType(n, 'error');
        }
        if (argTys.length !== sym.params!.length) {
          error(n, `\`${sym.name}\` takes ${sym.params!.length} argument(s) but ${argTys.length} supplied`, 'E0061');
        } else {
          argTys.forEach((t: Ty, i: number) => {
            if (t !== 'error' && t !== sym.params![i])
              error(n.args[i], `mismatched types: expected ${sym.params![i]}, found ${t}`, 'E0308');
          });
        }
        return setType(n, sym.ret!);
      }
      default:
        throw new Error(`sema: unexpected expression node ${n.kind}`);
    }
  }

  function block(b: AstNode, label: string): Ty {
    pushScope(b, label);
    for (const s of b.stmts) stmt(s);
    const ty = b.tail ? expr(b.tail) : 'unit';
    popScope();
    return setType(b, ty);
  }

  function stmt(s: AstNode): void {
    switch (s.kind) {
      case 'Let': {
        // The initializer is checked BEFORE the name is defined, so `let x = x;` sees the outer x.
        const t = expr(s.init);
        if (t !== 'error' && t !== s.ty) error(s.init, `mismatched types: expected ${s.ty}, found ${t}`, 'E0308');
        define(s, s.name, 'local', s.ty as Ty, s.mut);
        setType(s, 'unit');
        break;
      }
      case 'Assign': {
        const t = expr(s.value);
        const sym = resolve(s.target, s.target.name);
        if (sym) {
          setType(s.target, sym.ty);
          // The one visibly Rust-flavoured rule: bindings are immutable unless declared `mut`.
          if (!sym.mut) error(s, `cannot assign twice to immutable variable \`${sym.name}\``, 'E0384');
          if (t !== 'error' && t !== sym.ty)
            error(s.value, `mismatched types: expected ${sym.ty}, found ${t}`, 'E0308');
        }
        setType(s, 'unit');
        break;
      }
      case 'While': {
        const t = expr(s.cond);
        if (t !== 'error' && t !== 'bool') error(s.cond, `mismatched types: expected bool, found ${t}`, 'E0308');
        block(s.body, 'while body');
        setType(s, 'unit');
        break;
      }
      case 'If': {
        const t = expr(s.cond);
        if (t !== 'error' && t !== 'bool') error(s.cond, `mismatched types: expected bool, found ${t}`, 'E0308');
        block(s.then, 'if then');
        if (s.else) block(s.else, 'if else');
        setType(s, 'unit');
        break;
      }
      case 'Return': {
        const t = s.value ? expr(s.value) : 'unit';
        if (t !== 'error' && t !== currentFn.ret)
          error(s, `mismatched types: expected ${currentFn.ret}, found ${t}`, 'E0308');
        setType(s, 'unit');
        break;
      }
      case 'ExprStmt':
        expr(s.expr);
        setType(s, 'unit');
        break;
      default:
        throw new Error(`sema: unexpected statement node ${s.kind}`);
    }
  }

  // Global scope: every fn is visible to the whole file (hoisted), so order of definition doesn't matter.
  pushScope(root, 'global');
  for (const f of root.items) {
    if (scopes[0].names.has(f.name)) {
      error(f, `the name \`${f.name}\` is defined multiple times`, 'E0428');
      continue;
    }
    const sym = define(f, f.name, 'fn', f.ret, false);
    sym.params = f.params.map((p: AstNode) => p.ty as Ty);
    sym.ret = f.ret;
    sym.extern = f.isExtern;
  }
  for (const f of root.items) {
    currentFn = f;
    if (f.isExtern) {
      setType(f, f.ret);
      continue;
    }
    pushScope(f, `fn ${f.name}`);
    for (const p of f.params) {
      define(p, p.name, 'param', p.ty, false);
      setType(p, p.ty);
    }
    // The body block shares the parameter scope's lifetime but gets its own scope, like Rust.
    const bt = block(f.body, 'block');
    const last = f.body.stmts.at(-1);
    if (f.ret !== 'unit') {
      if (f.body.tail) {
        if (bt !== 'error' && bt !== f.ret)
          error(f.body.tail, `mismatched types: expected ${f.ret}, found ${bt}`, 'E0308');
      } else if (last?.kind !== 'Return')
        error(f.body, `mismatched types: function \`${f.name}\` must return ${f.ret}`, 'E0308');
    }
    popScope();
    setType(f, f.ret);
  }
  popScope();
  setType(root, 'unit');

  return { ok: errors.length === 0, errors, root, symbols, events };
}
