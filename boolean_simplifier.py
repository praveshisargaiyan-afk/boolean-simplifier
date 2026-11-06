"""
Boolean Expression Simplification - Minor Project
Quine-McCluskey + Petrick's method, K-map, truth table, Streamlit dashboard.

Run dashboard :  pip install streamlit pandas  &&  streamlit run boolean_simplifier.py
Run CLI       :  python boolean_simplifier.py "A&B | A&!B&C | !A&B"
Operators     :  NOT: ! ~ or postfix '   AND: & * AND   OR: | + OR   XOR: ^ XOR   () 0 1
"""
import re
import sys
from itertools import product, combinations

# ---------------------------------------------------------------- 1. PARSER
TOKEN = re.compile(r"\s*(?:([A-Za-z_]\w*)|(.))")
WORDS = {"AND": "&", "OR": "|", "NOT": "!", "XOR": "^"}


def tokenize(text):
    out = []
    for name, sym in TOKEN.findall(text):
        if name:
            up = name.upper()
            if up in WORDS:
                out.append(WORDS[up])
            elif name.isalpha() and len(name) > 1:
                out.extend(name)          # AB'C -> A B ' C (implicit AND); use A1, X_in for long names
            else:
                out.append(name)
        elif sym.strip():
            out.append(sym)
    return out


class Parser:
    """Grammar: or -> xor ('|' xor)* ; xor -> and ('^' and)* ; and -> not ('&' not)*
       not -> ('!'|'~') not | atom ("'")* ; atom -> VAR | 0 | 1 | '(' or ')'"""

    def __init__(self, text):
        self.t, self.i, self.vars = tokenize(text), 0, set()

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def eat(self, x=None):
        tok = self.peek()
        if tok is None or (x and tok != x):
            raise ValueError(f"Syntax error near token {self.i}: expected {x!r}, got {tok!r}")
        self.i += 1
        return tok

    def parse(self):
        node = self.p_or()
        if self.peek() is not None:
            raise ValueError(f"Unexpected token {self.peek()!r}")
        return node

    def p_or(self):
        n = self.p_xor()
        while self.peek() in ("|", "+"):
            self.eat(); n = ("or", n, self.p_xor())
        return n

    def p_xor(self):
        n = self.p_and()
        while self.peek() == "^":
            self.eat(); n = ("xor", n, self.p_and())
        return n

    def p_and(self):
        n = self.p_not()
        while self.peek() in ("&", "*") or (self.peek() not in (None, "|", "+", "^", ")")):
            if self.peek() in ("&", "*"):
                self.eat()
            n = ("and", n, self.p_not())          # implicit AND: AB'C
        return n

    def p_not(self):
        if self.peek() in ("!", "~"):
            self.eat(); return ("not", self.p_not())
        n = self.p_atom()
        while self.peek() == "'":
            self.eat(); n = ("not", n)
        return n

    def p_atom(self):
        tok = self.eat()
        if tok == "(":
            n = self.p_or(); self.eat(")"); return n
        if tok in ("0", "1"):
            return ("const", int(tok))
        if re.fullmatch(r"[A-Za-z_]\w*", tok):
            self.vars.add(tok); return ("var", tok)
        raise ValueError(f"Bad token {tok!r}")


def evaluate(node, env):
    k = node[0]
    if k == "const": return node[1]
    if k == "var":   return env[node[1]]
    if k == "not":   return 1 - evaluate(node[1], env)
    a, b = evaluate(node[1], env), evaluate(node[2], env)
    return a & b if k == "and" else a | b if k == "or" else a ^ b


def parse_expression(text):
    p = Parser(text)
    tree = p.parse()
    return tree, sorted(p.vars)


# ----------------------------------------------------------- 2. TRUTH TABLE
def truth_table(tree, variables):
    rows = []
    for bits in product((0, 1), repeat=len(variables)):
        rows.append((bits, evaluate(tree, dict(zip(variables, bits)))))
    return rows


# -------------------------------------------------- 3. QUINE-McCLUSKEY (QM)
def _ones(p): return p.count("1")


def combine(a, b):
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(diff) == 1 and "-" not in (a[diff[0]], b[diff[0]]):
        i = diff[0]
        return a[:i] + "-" + a[i + 1:]
    return None


def covers(pattern, m, n):
    s = format(m, f"0{n}b")
    return all(p in ("-", c) for p, c in zip(pattern, s))


def quine_mccluskey(minterms, dontcares, n):
    """Returns (prime_implicants, steps). steps = list of columns for display."""
    current = {format(m, f"0{n}b") for m in set(minterms) | set(dontcares)}
    primes, steps = set(), []
    while current:
        steps.append(sorted(current, key=lambda p: (_ones(p), p)))
        used, nxt = set(), set()
        for a, b in combinations(sorted(current), 2):
            c = combine(a, b)
            if c:
                nxt.add(c); used.update((a, b))
        primes |= current - used
        current = nxt
    return sorted(primes), steps


def cost(patterns):
    return (len(patterns), sum(len(p) - p.count("-") for p in patterns))


def select_cover(primes, minterms, n):
    """Essential PIs + Petrick's method for the rest. Returns (essentials, chosen, petrick_info)."""
    table = {m: [p for p in primes if covers(p, m, n)] for m in minterms}
    essentials = {ps[0] for ps in table.values() if len(ps) == 1}
    left = [m for m in minterms if not any(covers(e, m, n) for e in essentials)]
    if not left:
        return sorted(essentials), sorted(essentials), []
    # Petrick: product of sums -> sum of products with absorption
    prod = [frozenset([p]) for p in table[left[0]]]
    for m in left[1:]:
        new = {x | frozenset([p]) for x in prod for p in table[m]}
        prod = [s for s in new if not any(o < s for o in new)]      # absorption
    best = min(prod, key=lambda s: cost(s))
    return sorted(essentials), sorted(essentials | set(best)), [sorted(s) for s in prod]


def pattern_to_term(p, variables):
    lits = [(v if c == "1" else v + "'") for v, c in zip(variables, p) if c != "-"]
    if not lits:
        return "1"
    return "".join(lits) if all(len(v) == 1 for v in variables) else " & ".join(lits)


def simplify(minterms, dontcares, variables):
    n = len(variables)
    if not minterms:
        return {"expr": "0", "primes": [], "essentials": [], "chosen": [], "steps": [], "petrick": []}
    if len(set(minterms) | set(dontcares)) == 2 ** n and not [m for m in range(2**n) if m not in set(minterms) | set(dontcares)]:
        return {"expr": "1", "primes": ["-" * n], "essentials": ["-" * n], "chosen": ["-" * n], "steps": [], "petrick": []}
    primes, steps = quine_mccluskey(minterms, dontcares, n)
    ess, chosen, pet = select_cover(primes, minterms, n)
    expr = " + ".join(pattern_to_term(p, variables) for p in chosen)
    return {"expr": expr, "primes": primes, "essentials": ess, "chosen": chosen,
            "steps": steps, "petrick": pet}


def verify(minterms, dontcares, chosen, n):
    """Simplified SOP must be 1 on minterms, 0 on all maxterms (don't cares free)."""
    for m in range(2 ** n):
        val = any(covers(p, m, n) for p in chosen)
        if m in minterms and not val: return False
        if m not in minterms and m not in dontcares and val: return False
    return True


def literal_count(expr_patterns):
    return sum(len(p) - p.count("-") for p in expr_patterns)


# ----------------------------------------------------------------- 4. K-MAP
GRAY = {1: [0, 1], 2: [0, 1, 3, 2]}


def kmap(minterms, dontcares, variables):
    n = len(variables)
    if not 2 <= n <= 4:
        return None
    rv = n // 2
    cv = n - rv
    rows = GRAY.get(rv, [0, 1, 3, 2]); cols = GRAY.get(cv, [0, 1, 3, 2])
    g = lambda x, w: format(x, f"0{w}b")
    grid = []
    for r in rows:
        line = []
        for c in cols:
            m = (r << cv) | c
            line.append("X" if m in dontcares else "1" if m in minterms else "0")
        grid.append(line)
    return {"row_label": "".join(variables[:rv]), "col_label": "".join(variables[rv:]),
            "rows": [g(r, rv) for r in rows], "cols": [g(c, cv) for c in cols], "grid": grid}


# ---------------------------------------------------------------- 5. CLI
def run_cli(text):
    tree, variables = parse_expression(text)
    tt = truth_table(tree, variables)
    n = len(variables)
    minterms = [i for i, (_, o) in enumerate(tt) if o]
    res = simplify(minterms, [], variables)
    print("Variables :", variables)
    print("Minterms  : m", minterms)
    print("Primes    :", res["primes"])
    print("Essential :", res["essentials"])
    print("Simplified:", res["expr"])
    print("Verified  :", verify(set(minterms), set(), res["chosen"], n))


# ---------------------------------------------------------- 6. DASHBOARD
def dashboard():
    import streamlit as st
    import pandas as pd

    st.set_page_config(page_title="Boolean Simplifier", layout="wide")
    st.title("Boolean Expression Simplifier")
    st.caption("Quine-McCluskey + Petrick's method | up to 8 variables | don't-care support")

    examples = {
        "Redundant 3-var": "A&B | A&!B&C | !A&B",
        "Consensus": "AB + A'C + BC",
        "4-var": "A'B'C'D' + A'B'CD' + A'BC'D + ABC'D' + ABCD + AB'C'D'",
        "XOR": "A ^ B ^ C",
    }
    with st.sidebar:
        st.header("Input")
        pick = st.selectbox("Examples", list(examples))
        expr = st.text_input("Expression", examples[pick])
        st.markdown("`! ~ '` NOT, `& * AND` AND, `| + OR` OR, `^` XOR")
        dc_text = st.text_input("Don't-care minterms (comma separated)", "")

    try:
        tree, variables = parse_expression(expr)
        if not variables:
            st.error("No variables found."); return
        if len(variables) > 8:
            st.error("Maximum 8 variables supported."); return
        n = len(variables)
        tt = truth_table(tree, variables)
        dcs = {int(x) for x in dc_text.split(",") if x.strip().isdigit() and int(x) < 2 ** n}
        minterms = [i for i, (_, o) in enumerate(tt) if o and i not in dcs]
    except Exception as e:
        st.error(f"Could not parse expression: {e}"); return

    res = simplify(minterms, sorted(dcs), variables)
    ok = verify(set(minterms), dcs, res["chosen"], n) if res["chosen"] else True
    orig_lits = sum(1 for t in tokenize(expr) if re.fullmatch(r"[A-Za-z_]\w*", t))
    new_lits = literal_count(res["chosen"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Variables", n)
    c2.metric("Minterms", len(minterms))
    c3.metric("Literals", f"{orig_lits} -> {new_lits}", f"{new_lits - orig_lits}", delta_color="inverse")
    c4.metric("Verified", "PASS" if ok else "FAIL")
    st.success(f"Simplified SOP:  F = {res['expr']}")

    t1, t2, t3, t4, t5 = st.tabs(["Truth table", "QM steps", "Prime implicants", "K-map", "Gate count"])
    with t1:
        df = pd.DataFrame([list(b) + [o] for b, o in tt], columns=variables + ["F"])
        df.insert(0, "m", range(len(df)))
        st.dataframe(df, use_container_width=True, hide_index=True)
    with t2:
        for i, col in enumerate(res["steps"]):
            st.markdown(f"**Column {i + 1}**")
            st.code("  ".join(col))
    with t3:
        rows = []
        for p in res["primes"]:
            rows.append({"Implicant": p, "Term": pattern_to_term(p, variables),
                         "Essential": p in res["essentials"], "In final cover": p in res["chosen"],
                         "Covers": [m for m in minterms if covers(p, m, n)]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if res["petrick"]:
            st.markdown("**Petrick's method candidate covers**")
            for s in res["petrick"]:
                st.code(" + ".join(pattern_to_term(p, variables) for p in s))
    with t4:
        km = kmap(set(minterms), dcs, variables)
        if km:
            kdf = pd.DataFrame(km["grid"], index=km["rows"], columns=km["cols"])
            st.caption(f"Rows: {km['row_label']}   Columns: {km['col_label']}   (Gray code, X = don't care)")
            st.table(kdf)
        else:
            st.info("K-map is shown for 2-4 variables.")
    with t5:
        and_gates = sum(1 for p in res["chosen"] if len(p) - p.count("-") > 1)
        or_gate = 1 if len(res["chosen"]) > 1 else 0
        inv = len({v for p in res["chosen"] for v, c in zip(variables, p) if c == "0"})
        st.write({"AND gates": and_gates, "OR gates": or_gate, "NOT gates": inv,
                  "Product terms": len(res["chosen"]), "Literals": new_lits})


def _in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _in_streamlit():
        dashboard()                      # streamlit run boolean_simplifier.py
    elif len(sys.argv) > 1:
        run_cli(" ".join(sys.argv[1:]))  # python boolean_simplifier.py "A&B | !A&C"
    else:
        print('Use:  streamlit run boolean_simplifier.py   or   python boolean_simplifier.py "A&B | !A&C"')
