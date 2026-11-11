"""Look of the demo app: a watermelon palette on a flat, warm, paper-like base.

  flesh   watermelon red, used for FAKE and for primary actions
  rind    deep green, used for REAL and for structure (header, active tab, links)
  pith    cream and warm greys, the page and card surfaces
  seed    near-black ink for text and outlines

No gradients behind content, no glass effects: solid surfaces, 1.5 px outlines and a small offset
shadow, so cards read like printed stickers on paper.
"""

from __future__ import annotations

import gradio as gr

FLESH = gr.themes.Color(c50="#fff1f1", c100="#ffe1e1", c200="#ffc6c8", c300="#fb9da3", c400="#f2727d",
                        c500="#e4485b", c600="#cc3048", c700="#a9243a", c800="#8b2035", c900="#741e31",
                        c950="#410b17", name="flesh")
RIND = gr.themes.Color(c50="#f1f7f1", c100="#ddeedd", c200="#bcdcbd", c300="#8fc391", c400="#5fa564",
                       c500="#3f8846", c600="#2f6d36", c700="#27572d", c800="#224628", c900="#1c3a22",
                       c950="#0b1f10", name="rind")
PITH = gr.themes.Color(c50="#fcfaf5", c100="#f6f1e7", c200="#ebe3d3", c300="#d9cfbb", c400="#aba08a",
                       c500="#807562", c600="#62594a", c700="#4b443a", c800="#342f28", c900="#231f1a",
                       c950="#15120f", name="pith")

COLORS = {"fake": "#cc3048", "real": "#2f6d36", "unsure": "#b7791f", "ink": "#1f1b16",
          "muted": "#6b6253", "line": "#d9cfbb"}

THEME = gr.themes.Base(
    primary_hue=FLESH, secondary_hue=RIND, neutral_hue=PITH,
    radius_size=gr.themes.sizes.radius_md, spacing_size=gr.themes.sizes.spacing_md,
    font=[gr.themes.GoogleFont("Work Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "Menlo", "monospace"],
).set(
    body_background_fill="#f6f1e7", body_text_color="#1f1b16", body_text_color_subdued="#6b6253",
    background_fill_primary="#fffdf8", background_fill_secondary="#fcfaf5",
    block_background_fill="#fffdf8", block_border_color="#d9cfbb", block_border_width="1.5px",
    block_radius="10px", block_shadow="none", block_label_background_fill="#f6f1e7",
    block_label_text_color="#4b443a", block_title_text_color="#1f1b16", block_title_text_weight="600",
    border_color_primary="#d9cfbb", border_color_accent="#2f6d36",
    input_background_fill="#fffdf8", input_border_color="#d9cfbb", input_border_width="1.5px",
    input_border_color_focus="#2f6d36", input_shadow_focus="0 0 0 3px rgba(47,109,54,0.18)",
    button_primary_background_fill="#cc3048", button_primary_background_fill_hover="#a9243a",
    button_primary_text_color="#ffffff", button_primary_border_color="#1f1b16",
    button_secondary_background_fill="#fffdf8", button_secondary_background_fill_hover="#ddeedd",
    button_secondary_text_color="#1c3a22", button_secondary_border_color="#1f1b16",
    button_border_width="1.5px", button_large_radius="8px", button_small_radius="6px",
    slider_color="#cc3048", checkbox_label_background_fill_selected="#ddeedd",
    color_accent="#2f6d36", color_accent_soft="#ddeedd", link_text_color="#2f6d36",
    link_text_color_hover="#a9243a", table_even_background_fill="#fcfaf5", table_odd_background_fill="#fffdf8",
)

CSS = """
.gradio-container {max-width: 1240px !important; margin: 0 auto !important;}
.gradio-container h1, .gradio-container h2, .gradio-container h3, .wm-serif
  {font-family: 'Fraunces', Georgia, 'Times New Roman', serif; letter-spacing: -0.01em;}
footer {display: none !important;}

/* ---------- header ---------- */
.wm-top {display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 18px;
  padding: 18px 22px; margin-bottom: 6px; background: #1c3a22; color: #f6f1e7; border-radius: 14px;
  border: 1.5px solid #1f1b16; box-shadow: 4px 4px 0 #cc3048;}
.wm-top h1 {margin: 0; font-size: 1.55rem; font-weight: 600; color: #fffdf8;}
.wm-top p {margin: 4px 0 0; color: #c9dcc8; font-size: 0.92rem; max-width: 640px; line-height: 1.45;}
.wm-logo {width: 54px; height: 54px; flex: none;}
.wm-status {display: flex; flex-direction: column; gap: 6px; align-items: flex-end;}
.wm-pill {display: inline-flex; align-items: center; gap: 7px; padding: 4px 11px; border-radius: 6px;
  font-size: 0.78rem; font-weight: 500; background: rgba(255,253,248,0.08); color: #f6f1e7;
  border: 1px solid rgba(255,253,248,0.22); white-space: nowrap;}
.wm-led {width: 8px; height: 8px; border-radius: 2px; display: inline-block;}
.wm-led.on {background: #8fc391;} .wm-led.off {background: #807562;}

.wm-strip {display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 12px 0 4px;}
.wm-stat {background: #fffdf8; border: 1.5px solid #d9cfbb; border-radius: 10px; padding: 10px 14px;}
.wm-stat b {display: block; font-family: 'Fraunces', Georgia, serif; font-size: 1.45rem; font-weight: 600;
  font-variant-numeric: tabular-nums; color: #1f1b16; line-height: 1.15;}
.wm-stat span {font-size: 0.76rem; color: #6b6253; text-transform: uppercase; letter-spacing: 0.06em;}
.wm-stat small {display: block; font-size: 0.78rem; color: #6b6253; margin-top: 2px;}

/* ---------- tabs ---------- */
.wm-tabs > .tab-wrapper button, .wm-tabs button[role="tab"] {font-weight: 600 !important; font-size: 0.93rem !important;}
.wm-tabs button[role="tab"][aria-selected="true"] {color: #1c3a22 !important;}

/* ---------- cards and sections ---------- */
.wm-card {background: #fffdf8; border: 1.5px solid #1f1b16; border-radius: 12px; padding: 18px 20px;
  box-shadow: 3px 3px 0 #d9cfbb;}
.wm-card + .wm-card {margin-top: 12px;}
.wm-plain {background: #fffdf8; border: 1.5px solid #d9cfbb; border-radius: 12px; padding: 16px 18px;}
.wm-kicker {display: flex; align-items: center; gap: 9px; font-size: 0.75rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.08em; color: #6b6253; margin: 0 0 10px;}
.wm-kicker i {font-style: normal; display: inline-grid; place-items: center; width: 22px; height: 22px;
  border-radius: 6px; background: #1c3a22; color: #fffdf8; font-size: 0.75rem; letter-spacing: 0;}
.wm-h {font-family: 'Fraunces', Georgia, serif; font-size: 1.2rem; font-weight: 600; margin: 18px 0 4px; color: #1f1b16;}
.wm-lead {color: #6b6253; font-size: 0.9rem; margin: 0 0 10px; line-height: 1.5;}
.wm-note {color: #6b6253; font-size: 0.84rem; line-height: 1.45;}
.wm-input-card {background: #fffdf8 !important; border: 1.5px solid #1f1b16 !important; border-radius: 12px !important;
  box-shadow: 3px 3px 0 #d9cfbb !important; padding: 14px !important;}

/* ---------- empty state ---------- */
.wm-empty {text-align: center; padding: 34px 20px 30px;}
.wm-empty svg {width: 64px; height: 64px; margin-bottom: 8px;}
.wm-empty b {display: block; font-family: 'Fraunces', Georgia, serif; font-size: 1.2rem; margin-bottom: 4px;}
.wm-empty p {color: #6b6253; margin: 0 auto; max-width: 380px; font-size: 0.9rem;}
.wm-steps {display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 18px; text-align: left;}
.wm-steps div {border: 1.5px dashed #d9cfbb; border-radius: 8px; padding: 9px 10px; font-size: 0.82rem; color: #4b443a;}
.wm-steps em {display: block; font-style: normal; font-weight: 700; color: #cc3048; font-size: 0.72rem;
  letter-spacing: 0.08em; text-transform: uppercase;}

/* ---------- verdict ---------- */
.wm-verdict {display: grid; grid-template-columns: 1fr auto; gap: 14px; align-items: start;}
.wm-word {font-family: 'Fraunces', Georgia, serif; font-size: 2.3rem; font-weight: 700; line-height: 1; letter-spacing: 0.01em;}
.wm-label {margin-top: 6px; font-size: 0.95rem; color: #4b443a;}
.wm-num {text-align: right; font-family: 'Fraunces', Georgia, serif; font-size: 2.3rem; font-weight: 600;
  line-height: 1; font-variant-numeric: tabular-nums;}
.wm-num small {display: block; font-family: 'Work Sans', sans-serif; font-size: 0.74rem; font-weight: 500;
  color: #6b6253; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.06em;}
.wm-tags {display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px;}
.wm-tag {display: inline-flex; align-items: center; gap: 6px; padding: 3px 9px; border-radius: 6px;
  font-size: 0.78rem; font-weight: 600; border: 1.5px solid currentColor;}
.wm-tag.g {color: #2f6d36; background: #ddeedd;} .wm-tag.r {color: #a9243a; background: #ffe1e1;}
.wm-tag.y {color: #8a5a12; background: #fbefd5;} .wm-tag.n {color: #4b443a; background: #f6f1e7;}

.wm-meter {margin: 18px 0 4px;}
.wm-track {position: relative; height: 16px; display: grid; grid-template-columns: repeat(20, 1fr); gap: 3px;}
.wm-track span {border-radius: 3px; background: #ebe3d3;}
.wm-track span.lit-g {background: #5fa564;} .wm-track span.lit-r {background: #e4485b;}
.wm-mark {position: absolute; top: -7px; width: 0; height: 30px; border-left: 3px solid #1f1b16; transform: translateX(-1.5px);}
.wm-mark::after {content: ""; position: absolute; top: -6px; left: -7.5px; border: 6px solid transparent; border-top-color: #1f1b16;}
.wm-thr {position: absolute; top: -4px; height: 24px; border-left: 2px dashed #6b6253;}
.wm-scale {display: flex; justify-content: space-between; font-size: 0.74rem; color: #6b6253; margin-top: 6px;}
.wm-why {margin: 14px 0 0; padding: 0; list-style: none;}
.wm-why li {position: relative; padding: 5px 0 5px 20px; font-size: 0.92rem; line-height: 1.45; border-top: 1px dotted #d9cfbb;}
.wm-why li:first-child {border-top: none;}
.wm-why li::before {content: ""; position: absolute; left: 4px; top: 12px; width: 6px; height: 9px;
  border-radius: 50% 50% 50% 50% / 60% 60% 40% 40%; background: #1f1b16;}
.wm-truth {margin-top: 14px; padding: 10px 12px; border-radius: 8px; font-size: 0.88rem; line-height: 1.45;
  border: 1.5px solid;}
.wm-truth.ok {background: #f1f7f1; border-color: #5fa564;} .wm-truth.bad {background: #fff1f1; border-color: #f2727d;}
.wm-truth b {display: block; margin-bottom: 2px;}

/* ---------- evidence ---------- */
.wm-row {display: grid; grid-template-columns: 150px 1fr 70px; gap: 12px; align-items: center;
  padding: 10px 0; border-bottom: 1px solid #ebe3d3;}
.wm-row:last-of-type {border-bottom: none;}
.wm-row strong {display: block; font-size: 0.95rem;}
.wm-row small {display: block; font-size: 0.76rem; color: #6b6253; line-height: 1.3;}
.wm-bar {position: relative; height: 10px; border-radius: 3px; background: #ebe3d3;}
.wm-bar > span {position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px;}
.wm-bar > i {position: absolute; top: -4px; bottom: -4px; border-left: 2px dashed #6b6253;}
.wm-val {text-align: right; font-weight: 700; font-variant-numeric: tabular-nums;}
.wm-na {color: #807562; font-size: 0.85rem;}
.wm-dl {display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; margin: 14px 0 0;}
.wm-dl div {background: #fcfaf5; border: 1px solid #ebe3d3; border-radius: 8px; padding: 8px 10px;}
.wm-dl dt {font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: #6b6253;}
.wm-dl dd {margin: 2px 0 0; font-weight: 600; font-size: 0.9rem; overflow-wrap: anywhere;}
.wm-notes {margin-top: 14px; border-left: 4px solid #b7791f; background: #fdf6e6; border-radius: 0 8px 8px 0; padding: 10px 14px;}
.wm-notes b {font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; color: #8a5a12;}
.wm-notes ol {margin: 6px 0 0; padding-left: 20px;} .wm-notes li {font-size: 0.88rem; line-height: 1.45; margin: 4px 0;}

/* ---------- tables ---------- */
.wm-table-wrap {overflow-x: auto; border: 1.5px solid #d9cfbb; border-radius: 10px; background: #fffdf8;}
table.wm-table {width: 100%; border-collapse: collapse !important; font-size: 0.88rem; margin: 0 !important; border: none !important;}
table.wm-table th {text-align: left; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: #6b6253; font-weight: 600; background: #f6f1e7 !important; padding: 9px 12px !important; border: none !important;
  border-bottom: 1.5px solid #d9cfbb !important; white-space: nowrap;}
table.wm-table td {padding: 9px 12px !important; border: none !important; border-bottom: 1px solid #ebe3d3 !important;
  vertical-align: middle; background: transparent !important;}
table.wm-table tr:last-child td {border-bottom: none !important;}
table.wm-table td.num, table.wm-table th.num {text-align: right; font-variant-numeric: tabular-nums;}
.wm-chip {display: inline-block; padding: 2px 8px; border-radius: 5px; font-size: 0.76rem; font-weight: 700; letter-spacing: 0.03em;}
.wm-chip.FAKE {background: #ffe1e1; color: #a9243a;} .wm-chip.REAL {background: #ddeedd; color: #27572d;}
.wm-chip.INCONCLUSIVE {background: #fbefd5; color: #8a5a12;}
.wm-yes {color: #2f6d36; font-weight: 700;} .wm-no {color: #cc3048; font-weight: 700;}
.wm-mini {display: inline-block; width: 70px; height: 7px; border-radius: 2px; background: #ebe3d3; vertical-align: middle; margin-right: 8px; position: relative; overflow: hidden;}
.wm-mini > span {position: absolute; left: 0; top: 0; bottom: 0;}

/* ---------- models ---------- */
.wm-models {display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px;}
.wm-models h3 {margin: 0 0 4px; font-size: 1.15rem;}
.wm-badge {display: inline-block; font-family: 'Work Sans', sans-serif; padding: 2px 8px; border-radius: 5px;
  font-size: 0.7rem; font-weight: 700; margin-left: 6px; vertical-align: middle; text-transform: uppercase; letter-spacing: 0.05em;}
.wm-badge.own {background: #ffe1e1; color: #a9243a;} .wm-badge.ext {background: #ddeedd; color: #27572d;}
.wm-spec {display: grid; grid-template-columns: 120px 1fr; gap: 0; margin: 12px 0 0; font-size: 0.88rem;}
.wm-spec dt, .wm-spec dd {padding: 7px 0; border-top: 1px solid #ebe3d3; margin: 0;}
.wm-spec dt {color: #6b6253;}
.wm-tiles {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;}
.wm-tile {background: #fffdf8; border: 1.5px solid #d9cfbb; border-radius: 10px; padding: 12px 14px;}
.wm-tile span {display: block; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: #6b6253;}
.wm-tile b {display: block; font-family: 'Fraunces', Georgia, serif; font-size: 1.5rem; font-weight: 600; font-variant-numeric: tabular-nums;}
.wm-tile small {display: block; font-size: 0.76rem; color: #6b6253;}
.wm-cm {display: grid; grid-template-columns: auto 1fr 1fr; gap: 6px; max-width: 420px; font-variant-numeric: tabular-nums;}
.wm-cm div {padding: 10px 12px; border-radius: 8px; text-align: center; font-weight: 700;}
.wm-cm .h {background: transparent; font-weight: 600; font-size: 0.76rem; color: #6b6253; text-align: left;}
.wm-cm .ok {background: #ddeedd; color: #1c3a22;} .wm-cm .bad {background: #ffe1e1; color: #8b2035;}

/* ---------- code ---------- */
.wm-code-head {display: grid; grid-template-columns: 34px 1fr; gap: 12px; align-items: start; margin: 22px 0 8px;}
.wm-code-head > i {font-style: normal; display: grid; place-items: center; width: 34px; height: 34px; border-radius: 8px;
  background: #cc3048; color: #fffdf8; font-weight: 700; border: 1.5px solid #1f1b16;}
.wm-code-head b {display: block; font-size: 1rem;}
.wm-code-head p {margin: 2px 0 0; color: #6b6253; font-size: 0.86rem; line-height: 1.45;}
.wm-code-head code {font-size: 0.76rem; background: #f6f1e7; padding: 1px 6px; border-radius: 4px; color: #4b443a;}
.wm-group {font-family: 'Fraunces', Georgia, serif; font-size: 1.3rem; font-weight: 600; margin: 26px 0 0;
  padding-bottom: 6px; border-bottom: 2px solid #1f1b16;}

/* ---------- method ---------- */
.wm-grid2 {display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 14px;}
.wm-pipe {list-style: none; margin: 8px 0 0; padding: 0; counter-reset: s;}
.wm-pipe li {counter-increment: s; position: relative; padding: 9px 0 9px 40px; border-top: 1px solid #ebe3d3; font-size: 0.9rem; line-height: 1.45;}
.wm-pipe li:first-child {border-top: none;}
.wm-pipe li::before {content: counter(s); position: absolute; left: 0; top: 8px; width: 26px; height: 26px; border-radius: 6px;
  display: grid; place-items: center; background: #ddeedd; color: #1c3a22; font-weight: 700; font-size: 0.8rem;}
.wm-list {list-style: none; margin: 8px 0 0; padding: 0;}
.wm-list li {position: relative; padding: 7px 0 7px 22px; border-top: 1px dotted #d9cfbb; font-size: 0.9rem; line-height: 1.45;}
.wm-list li:first-child {border-top: none;}
.wm-list li::before {content: ""; position: absolute; left: 5px; top: 13px; width: 6px; height: 9px;
  border-radius: 50% 50% 50% 50% / 60% 60% 40% 40%; background: #1f1b16;}
.wm-list.warn li::before {background: #cc3048;}

@media (max-width: 860px) {
  .wm-top {grid-template-columns: auto 1fr;} .wm-status {grid-column: 1 / -1; flex-direction: row; flex-wrap: wrap; align-items: flex-start;}
  .wm-strip {grid-template-columns: repeat(2, minmax(0, 1fr));}
  .tab-wrapper, .tab-container {min-width: 0 !important; max-width: 100% !important;}
  .tab-container {flex-wrap: wrap !important; height: auto !important; row-gap: 2px;}
  .wm-stack {flex-direction: column !important; flex-wrap: nowrap !important;}
  .wm-stack > * {min-width: 0 !important; width: 100% !important; flex: 1 1 auto !important;}
}
@media (max-width: 560px) {
  .wm-top {grid-template-columns: 1fr; padding: 16px;} .wm-top .wm-logo {width: 40px; height: 40px;}
  .wm-top h1 {font-size: 1.3rem;}
  .wm-word, .wm-num {font-size: 1.8rem;} .wm-row {grid-template-columns: 96px 1fr 56px; gap: 8px;}
  .wm-steps {grid-template-columns: 1fr;} .wm-spec {grid-template-columns: 1fr;} .wm-spec dd {border-top: none; padding-top: 0;}
}
"""

HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700'
        '&display=swap" rel="stylesheet">')

LOGO = ('<svg class="wm-logo" viewBox="0 0 64 64" aria-hidden="true">'
        '<path d="M4 22 A28 28 0 0 0 60 22 Z" fill="#3f8846" stroke="#1f1b16" stroke-width="2"/>'
        '<path d="M9 22 A23 23 0 0 0 55 22 Z" fill="#f6f1e7"/>'
        '<path d="M12 22 A20 20 0 0 0 52 22 Z" fill="#e4485b"/>'
        '<g fill="#1f1b16"><ellipse cx="24" cy="29" rx="1.6" ry="2.4"/><ellipse cx="32" cy="34" rx="1.6" ry="2.4"/>'
        '<ellipse cx="40" cy="29" rx="1.6" ry="2.4"/><ellipse cx="28" cy="38" rx="1.4" ry="2.1"/>'
        '<ellipse cx="36" cy="38" rx="1.4" ry="2.1"/></g>'
        '<line x1="4" y1="22" x2="60" y2="22" stroke="#1f1b16" stroke-width="2"/></svg>')
