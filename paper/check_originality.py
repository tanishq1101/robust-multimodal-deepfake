"""Local checks (not a substitute for Turnitin/iThenticate):
1. Word 6-gram overlap of the paper's own prose against the project's earlier documents
   (self-overlap) -- reference list and equations excluded.
2. A scan for words and phrases that commonly trigger AI-text suspicion.
"""
import re, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
tex = (Path(__file__).parent / "paper.tex").read_text()
body = tex.split("\\begin{abstract}")[1].split("\\bibliographystyle")[0]
body = re.sub(r"\\begin\{(equation|align|tikzpicture)\*?\}.*?\\end\{\1\*?\}", " ", body, flags=re.S)
body = re.sub(r"\\(cite|ref|eqref|label)\{[^}]*\}", " ", body)
body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", body)
body = re.sub(r"[{}$~\\]", " ", body)

def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())

def shingles(w, n=6):
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}

pw = words(body)
ps = shingles(pw)
print(f"paper prose: {len(pw)} words, {len(ps)} six-grams")
for name in ["reports/report.md", "reports/literature_review.md", "README.md", "PLAN.md", "reports/review.md", "TASKS.md"]:
    f = root / name
    if not f.exists():
        continue
    other = shingles(words(f.read_text()))
    shared = ps & other
    print(f"  {name:32s} shared 6-grams: {len(shared):3d}  ({100 * len(shared) / len(ps):.1f}% of paper)")
    if "-v" in sys.argv:
        for s in sorted(shared)[:15]:
            print("      ", " ".join(s))

flag = r"""\b(delve|delves|delving|leverag\w*|crucial\w*|pivotal|tapestry|landscape|realm|showcas\w*|underscor\w*|
comprehensive\w*|seamless\w*|notably|furthermore|moreover|additionally|in conclusion|it is worth noting|
it is important to note|holistic|cutting-edge|groundbreaking|game-chang\w*|paradigm shift|testament|
multifaceted|intricate|vibrant|foster\w*|harness\w*|elevate\w*|empower\w*|streamline\w*|robust\w*|
novel|state-of-the-art|plays? a (vital|key|crucial) role|in today's|ever-evolving|navigate\w*|
shed light|meticulous\w*|myriad|plethora|pave the way|unlock\w*|cornerstone|beacon|bolster\w*|
nuanced|underpin\w*|facilitat\w*|utiliz\w*|endeavou?r\w*|aforementioned|thereby)\b"""
flag = re.compile(flag.replace("\n", ""), re.I)
print("\nflagged words:")
for m in sorted({(x.group(0).lower()) for x in flag.finditer(body)}):
    print("  ", m, len(re.findall(re.escape(m), body, re.I)))
print("em dashes in prose:", body.count("---"), "| '--' as dash:", len(re.findall(r" -- ", body)))
sent = [s for s in re.split(r"(?<=[.!?])\s+", " ".join(body.split())) if len(s.split()) > 3]
lens = [len(s.split()) for s in sent]
import statistics as st
print(f"sentences: {len(lens)}, mean length {st.mean(lens):.1f}, stdev {st.pstdev(lens):.1f}, min {min(lens)}, max {max(lens)}")
