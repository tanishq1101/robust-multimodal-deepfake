"""Convert paper.tex (the restricted LaTeX subset it uses) into blocks.json for build_docx.cjs.

Keeps the Word file in sync with the LaTeX source: text, numbers, citations and reference list all
come from paper.tex, numbers.tex, attack_table.tex and paper.bbl. Display equations are rendered
to PNG with matplotlib's mathtext.
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
(HERE / "eq").mkdir(exist_ok=True)
tex = (PAPER / "paper.tex").read_text()
numbers = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (PAPER / "numbers.tex").read_text()))
bbl = (PAPER / "paper.bbl").read_text()

bib_keys = re.findall(r"\\bibitem\{([^}]+)\}", bbl)
cite_no = {k: i + 1 for i, k in enumerate(bib_keys)}

# ---------------------------------------------------------------- labels (figure / table / equation numbers)
fig_no, tab_no, eq_no = {}, {}, {}
roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
for env, store in (("figure\\*?", fig_no), ("table", tab_no)):
    for i, m in enumerate(re.finditer(r"\\begin\{(%s)\}(.*?)\\end\{\1\}" % env, tex, re.S)):
        lab = re.search(r"\\label\{([^}]+)\}", m.group(2))
        if lab:
            store[lab.group(1)] = i + 1
n = 0
for m in re.finditer(r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}", tex, re.S):
    if m.group(1) == "equation":
        n += 1
        lab = re.search(r"\\label\{([^}]+)\}", m.group(2))
        if lab:
            eq_no[lab.group(1)] = n
    else:
        n += len([x for x in m.group(2).split("\\\\") if x.strip()])

# ---------------------------------------------------------------- inline math -> runs
SYM = {"times": "×", "leq": "≤", "ell": "ℓ", "sigma": "σ", "lambda": "λ", "epsilon": "ε", "alpha": "α",
       "lor": "∨", "in": "∈", "top": "⊤", "dagger": "†", "cdot": "·", "max": "max", "mathcal": "",
       "sum": "Σ", "partial": "∂", "ge": "≥", "tilde": "~"}
ACCENT = {r'\"o': "ö", r'\"u': "ü", r'\"a': "ä", r'\ss': "ß", r"\v{s}": "š", r"\'e": "é"}


def parse_math(s, runs, style=None, depth=0):
    style = dict(style or {})
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([a-zA-Z]+)", s[i:])
            if m:
                name = m.group(1)
                i += len(m.group(0))
                if name in ("mathbf", "mathrm", "text", "mathbb", "mathcal"):
                    j = i
                    while j < len(s) and s[j] == " ":
                        j += 1
                    if s[j] == "{":
                        d, k = 1, j + 1
                        while d:
                            d += {"{": 1, "}": -1}.get(s[k], 0)
                            k += 1
                        arg = s[j + 1:k - 1]
                        i = k
                    else:
                        arg, i = s[j], j + 1
                    st = dict(style)
                    if name == "mathbf":
                        st["b"] = True
                        st["upright"] = True
                    elif name in ("mathrm", "text"):
                        st["upright"] = True
                    elif name == "mathbb":
                        arg = {"R": "ℝ"}.get(arg, arg)
                        st["upright"] = True
                    parse_math(arg, runs, st, depth + 1)
                elif name == "tilde":
                    j = i
                    while s[j] == " ":
                        j += 1
                    if s[j] == "{":
                        arg = s[j + 1:s.index("}", j)]
                        i = s.index("}", j) + 1
                    else:
                        arg, i = s[j], j + 1
                    runs.append(dict(t=arg + "\u0303", **style))
                else:
                    runs.append(dict(t=SYM.get(name, name), **{k: v for k, v in style.items() if k != "upright"},
                                     upright=True))
            else:
                nxt = s[i + 1]
                i += 2
                if nxt == ",":
                    runs.append(dict(t="\u2009", **style))
                elif nxt in "{}":
                    runs.append(dict(t=nxt, **style))
            continue
        if c in "_^":
            key = "sub" if c == "_" else "sup"
            i += 1
            if s[i] == "{":
                d, k = 1, i + 1
                while d:
                    d += {"{": 1, "}": -1}.get(s[k], 0)
                    k += 1
                arg = s[i + 1:k - 1]
                i = k
            elif s[i] == "\\":
                m = re.match(r"\\[a-zA-Z]+", s[i:])
                arg, i = m.group(0), i + len(m.group(0))
            else:
                arg, i = s[i], i + 1
            sub = []
            parse_math(arg, sub, {}, depth + 1)
            for r in sub:
                r[key] = True
                r.setdefault("upright", False)
                for k2, v2 in style.items():
                    r.setdefault(k2, v2)
            runs.extend(sub)
            continue
        if c == "{":
            d, k = 1, i + 1
            while d:
                d += {"{": 1, "}": -1}.get(s[k], 0)
                k += 1
            if s[i + 1:k - 1] == ",":
                runs.append(dict(t=",", **style))
            else:
                parse_math(s[i + 1:k - 1], runs, style, depth + 1)
            i = k
            continue
        if c == " ":
            i += 1
            continue
        j = i
        if c.isalpha():
            runs.append(dict(t=c, **style) if style.get("upright") else dict(t=c, i=True, **{k: v for k, v in style.items()}))
        else:
            runs.append(dict(t=c, **style))
        i += 1
    return runs


# ---------------------------------------------------------------- text -> runs
def tex_to_runs(s):
    s = s.replace("\n", " ")
    for k, v in numbers.items():
        s = re.sub(r"\\%s(?![a-zA-Z])(\{\})?" % k, v, s)

    def cite(m):
        ns = sorted(cite_no[k.strip()] for k in m.group(1).split(","))
        out, i = [], 0
        while i < len(ns):
            j = i
            while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
                j += 1
            if j - i >= 2:
                out.append(f"[{ns[i]}]–[{ns[j]}]")
            else:
                out.extend(f"[{x}]" for x in ns[i:j + 1])
            i = j + 1
        return "\u0001".join(out) if False else ", ".join(out)

    s = re.sub(r"\\cite\{([^}]*)\}", cite, s)
    s = re.sub(r"\\eqref\{([^}]*)\}", lambda m: f"({eq_no[m.group(1)]})", s)

    def ref(m):
        k = m.group(1)
        if k in fig_no:
            return str(fig_no[k])
        if k in tab_no:
            return roman[tab_no[k] - 1]
        return "?"

    s = re.sub(r"\\ref\{([^}]*)\}", ref, s)
    s = re.sub(r"\\IEEEPARstart\{(.)\}\{", lambda m: "{" + m.group(1), s)
    for k, v in ACCENT.items():
        s = s.replace("{" + k + "}", v)
    s = s.replace("``", "\u201c").replace("''", "\u201d").replace("---", "—").replace("--", "–")
    s = s.replace("\\%", "%").replace("\\&", "&").replace("~", "\u00a0").replace("\\,", "\u2009").replace("\\ ", " ")
    s = s.replace("\\_", "_").replace("{,}", ",").replace("\\dots", "…")
    s = re.sub(r"\\-", "", s)
    s = re.sub(r"\\url\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    # split into segments: math, emph, bold, texttt, plain
    runs = []
    pos = 0
    pat = re.compile(r"\$(.+?)\$|\\(emph|textbf|texttt)\{|\\multicolumn")
    stack = []  # open style braces: (style_key, brace_depth)
    out = []

    def add_plain(txt, st):
        txt = re.sub(r"\\[a-zA-Z]+\*?", "", txt) if "\\" in txt else txt
        txt = txt.replace("{", "").replace("}", "")
        if txt:
            out.append(dict(t=txt, **st))

    def walk(seg, st):
        p = 0
        for m in re.finditer(r"\$(.+?)\$|\\(emph|textbf|texttt)\{", seg):
            if m.start() < p:
                continue
            add_plain(seg[p:m.start()], st)
            if m.group(1) is not None:
                parse_out = parse_math(m.group(1), [], st)
                out.extend(parse_out)
                p = m.end()
            else:
                d, k = 1, m.end()
                while d:
                    d += {"{": 1, "}": -1}.get(seg[k], 0)
                    k += 1
                inner = seg[m.end():k - 1]
                key = {"emph": "i", "textbf": "b", "texttt": "code"}[m.group(2)]
                walk(inner, {**st, key: True})
                p = k
        add_plain(seg[p:], st)

    walk(s, {})
    # merge adjacent runs with identical styling
    merged = []
    for r in out:
        if merged and {k: v for k, v in merged[-1].items() if k != "t"} == {k: v for k, v in r.items() if k != "t"}:
            merged[-1]["t"] += r["t"]
        else:
            merged.append(r)
    for r in merged:
        r["t"] = re.sub(r" {2,}", " ", r["t"])
    return merged


# ---------------------------------------------------------------- equation images
plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "STIXGeneral"})


def render_eq(latex, name):
    s = latex
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    for a in ("\\bigg", "\\Bigg", "\\Big", "\\big"):
        s = s.replace(a + "l", a).replace(a + "r", a)
        s = s.replace(a + "[", "[").replace(a + "]", "]").replace(a + "(", "(").replace(a + ")", ")")
    s = re.sub(r"\\(bigg|Bigg|Big|big)[lr]?\s*\|", "|", s)
    s = s.replace("\\!", "").replace("\\,", r"\ ").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\text{", "\\mathrm{").replace("\\mathrm{real}", "\\mathrm{real}").replace("\\mathrm{fake}", "\\mathrm{fake}")
    s = s.replace("\\mathbf{W}", "\\mathbf{W}")
    s = s.replace("\\lor", "\\vee")
    s = s.replace("&", "")
    s = re.sub(r"\\(mathbf|mathcal|mathbb|mathrm)\s+([A-Za-z])", r"\\\1{\2}", s)
    s = re.sub(r"\\frac(\d)([A-Za-z])", r"\\frac{\1}{\2}", s)
    s = s.replace("\\qquad", "\\quad\\quad")
    s = re.sub(r"\s+", " ", s).strip().rstrip(",.")
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, f"${s}$", fontsize=10)
    path = HERE / "eq" / f"{name}.png"
    fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    w, h = Image.open(path).size
    return dict(path=str(path), w=w / 400 * 96, h=h / 400 * 96)


# ---------------------------------------------------------------- tabular
def parse_tabular(body):
    body = re.sub(r"\\input\{attack_table\.tex\}", lambda m: (PAPER / "attack_table.tex").read_text(), body)
    m = re.search(r"\\begin\{tabular\}\{(.*?)\}\s*\n(.*?)\\end\{tabular\}", body, re.S)
    spec, content = m.group(1), m.group(2)
    spec = re.sub(r"@\{\}", "", spec)
    spec = re.sub(r">\{\\raggedright\\arraybackslash\}", "", spec)
    aligns = []
    for ch in re.findall(r"p\{[^}]*\}|[lrc]", spec):
        aligns.append("l" if ch.startswith("p") else ch)
    rows, rules, note = [], [], None
    pending = None
    for line in re.split(r"\\\\", content):
        line = line.strip()
        marks = re.findall(r"\\(toprule|midrule|bottomrule)", line)
        line = re.sub(r"\\(toprule|midrule|bottomrule)", "", line).strip()
        if line.startswith("\\multicolumn"):
            mm = re.search(r"\\multicolumn\{\d+\}\{(?:[^{}]|\{[^{}]*\})*\}\{(.*)\}\s*$", line, re.S)
            note = tex_to_runs(re.sub(r"\\scriptsize\s*", "", mm.group(1)))
            if marks:
                pass
            continue
        for mk in marks:
            if mk == "midrule":
                pending = "mid"
        if mk_top := ("toprule" in marks):
            pass
        if not line:
            if "bottomrule" in marks and rows:
                rows[-1]["rule_after"] = True
            continue
        cells = [tex_to_runs(c.strip()) for c in line.split(" & ")] if " & " in line else [tex_to_runs(c.strip()) for c in line.split("&")]
        row = {"cells": cells, "rule_before": pending}
        pending = None
        rows.append(row)
        if "bottomrule" in marks:
            row["rule_after"] = True
    rows[-1]["rule_after"] = True
    return dict(aligns=aligns, rows=rows, note=note)


# ---------------------------------------------------------------- walk the document
body = tex.split("\\maketitle")[1].split("\\bibliographystyle")[0]
blocks = []

abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S).group(1)
keywords = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", body, re.S).group(1)
blocks.append(dict(type="abstract", runs=tex_to_runs(abstract)))
blocks.append(dict(type="keywords", runs=tex_to_runs(keywords)))
rest = body.split("\\end{IEEEkeywords}")[1]

pat = re.compile(r"\\section(\*?)\{(.*?)\}|\\subsection\{(.*?)\}|\\begin\{(figure\*?|table|equation|align|enumerate)\}(.*?)\\end\{\4\}", re.S)
sec_n = 0
sub_n = 0
eqn = 0
pos = 0


def add_text(txt):
    txt = re.sub(r"(?m)^%.*$", "", txt)
    for para in re.split(r"\n\s*\n", txt):
        para = para.strip()
        if para:
            blocks.append(dict(type="para", runs=tex_to_runs(para)))


for m in pat.finditer(rest):
    add_text(rest[pos:m.start()])
    pos = m.end()
    if m.group(2) is not None:
        if m.group(1) == "*":
            blocks.append(dict(type="h1", text=m.group(2).upper()))
        else:
            sec_n += 1
            sub_n = 0
            blocks.append(dict(type="h1", text=f"{roman[sec_n - 1]}. {m.group(2).upper()}"))
    elif m.group(3) is not None:
        sub_n += 1
        blocks.append(dict(type="h2", text=f"{chr(64 + sub_n)}. {m.group(3)}"))
    else:
        env, inner = m.group(4), m.group(5)
        if env == "enumerate":
            items = [i.strip() for i in re.split(r"\\item", inner) if i.strip()]
            blocks.append(dict(type="enum", items=[tex_to_runs(i) for i in items]))
        elif env == "equation":
            eqn += 1
            blocks.append(dict(type="eq", num=eqn, **render_eq(inner, f"eq{eqn}")))
        elif env == "align":
            for line in [x for x in inner.split("\\\\") if x.strip()]:
                eqn += 1
                blocks.append(dict(type="eq", num=eqn, **render_eq(line, f"eq{eqn}")))
        elif env == "table":
            cap = re.search(r"\\caption\{(.*?)\}\s*\\label", inner, re.S).group(1)
            num = [v for k, v in tab_no.items() if re.search(r"\\label\{%s\}" % k, inner)][0]
            blocks.append(dict(type="table", num=roman[num - 1], caption=cap.upper() if False else tex_to_runs(cap), **parse_tabular(inner)))
        else:  # figure / figure*
            cap = tex_to_runs(re.search(r"\\caption\{(.*)\}\s*\\label", inner, re.S).group(1))
            num = [v for k, v in fig_no.items() if re.search(r"\\label\{%s\}" % k, inner)][0]
            if env == "figure*":
                img, wide, frac = HERE / "arch.png", True, 1.0
            else:
                g = re.search(r"\\includegraphics\[width=([\d.]*)\\columnwidth\]\{(\w+)\}", inner)
                frac = float(g.group(1) or 1)
                img = PAPER / "figures" / (g.group(2) + ".png")
                wide = False
            w, h = Image.open(img).size
            width_in = 7.16 if wide else 3.5 * frac
            blocks.append(dict(type="figure", wide=wide, num=num, path=str(img), w=width_in * 96, h=width_in * 96 * h / w, caption=cap))
add_text(rest[pos:])

# references
refs = []
for m in re.finditer(r"\\bibitem\{([^}]+)\}\s*(.*?)(?=\n\s*\n|\\end\{thebibliography\})", bbl, re.S):
    txt = re.sub(r"\s+", " ", m.group(2))
    txt = txt.replace("\\newblock", "").replace("\\BIBentryALTinterwordspacing", "").replace("\\BIBentrySTDinterwordspacing", "")
    txt = re.sub(r"\\url\{([^}]*)\}", r"\1", txt)
    txt = re.sub(r"\\hskip\s*1em plus 0\.5em minus 0\.4em(\\relax)?", " ", txt)
    refs.append(tex_to_runs(txt))
assert len(refs) == len(bib_keys), (len(refs), len(bib_keys))

title = re.search(r"\\title\{(.*?)\}\s*\n", tex, re.S).group(1)
title = title.replace("~", " ")
data = dict(title=title, blocks=blocks, refs=refs)
(HERE / "blocks.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
print(len(blocks), "blocks;", len(refs), "refs;", eqn, "equations")
