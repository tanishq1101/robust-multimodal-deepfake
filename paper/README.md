# IEEE paper

Build (needs `tectonic`, installed with `brew install tectonic`; the repo `.venv` is used for the figures):

```bash
.venv/bin/python paper/make_figures.py   # figures + paper/analysis.json from models/audio_asvspoof_3a53cf51/
python3 paper/gen_tex.py                 # numbers.tex and attack_table.tex from analysis.json
cd paper && tectonic paper.tex           # paper.pdf (IEEEtran, runs BibTeX itself)
python3 paper/check_originality.py -v    # self-overlap and AI-phrasing scan
```

No number in `paper.tex` is typed by hand: results come from `analysis.json` via `numbers.tex`.
Author names and affiliation in `paper.tex` are placeholders.

Word version: `python3 paper/docx_build/tex_to_json.py` (use the repo `.venv` python for matplotlib), then
`node paper/docx_build/build_docx.cjs` (needs the `docx` npm package) writes `paper_word.docx`;
`soffice --headless --convert-to pdf paper_word.docx` gives `paper_word.pdf`.
Display equations in the Word file are images; everything else is editable text.
