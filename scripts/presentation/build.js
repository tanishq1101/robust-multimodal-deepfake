// Builds reports/presentation.pptx, the project review deck.
//
//   cd scripts/presentation && npm install && npm run build      (or: node build.js [out.pptx])
//
// Every number is read from the published run in models/audio_asvspoof_3a53cf51 (test metrics,
// per-attack results, history and the 5,000 test predictions for the ROC curve), so rebuilding
// after a new run picks up its results. Figures come from reports/figures/.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const JSZip = require("jszip");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const REPO = path.resolve(__dirname, "..", "..");
const RUN = path.join(REPO, "models", "audio_asvspoof_3a53cf51");
const OUT = path.resolve(process.argv[2] || path.join(REPO, "reports", "presentation.pptx"));

// ---------------------------------------------------------------- data
function readCsv(file) {
  const [header, ...rows] = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const keys = header.split(",");
  return rows.map((r) => Object.fromEntries(r.split(",").map((v, i) => [keys[i], v === "" ? null : isNaN(v) ? v : +v])));
}

// ROC from the saved predictions, resampled at fixed false-positive rates for a light chart.
function rocPoints(preds) {
  const pos = preds.filter((p) => p.label === 1).length, neg = preds.length - pos;
  const sorted = [...preds].sort((a, b) => b.score - a.score);
  const pts = [[0, 0]];
  let tp = 0, fp = 0;
  sorted.forEach((p, i) => {
    p.label === 1 ? tp++ : fp++;
    if (i === sorted.length - 1 || sorted[i + 1].score !== p.score) pts.push([fp / neg, tp / pos]);
  });
  const grid = [...new Set([...Array(21).keys()].map((i) => i * 0.001)
    .concat([...Array(19).keys()].map((i) => 0.02 + i * 0.01), [...Array(17).keys()].map((i) => 0.2 + i * 0.05))
    .map((x) => +x.toFixed(4)))].sort((a, b) => a - b);
  // highest TPR reached at or below each false-positive rate
  return [[0, 0], ...grid.filter((x) => x > 0).map((x) => [x, Math.max(...pts.filter((p) => p[0] <= x + 1e-12).map((p) => p[1]))])];
}

function loadData() {
  const metrics = JSON.parse(fs.readFileSync(path.join(RUN, "test", "metrics.json"), "utf8"));
  return {
    metrics,
    cats: readCsv(path.join(RUN, "test", "per_category.csv")),
    history: readCsv(path.join(RUN, "history.csv")),
    roc: rocPoints(readCsv(path.join(RUN, "test", "predictions.csv"))),
  };
}
const D = loadData();

// ---------------------------------------------------------------- theme
// pptxgenjs writes Office's stock colour scheme; replace it with THEME so that scheme colours
// (C.accent1, ...) and later restyling in PowerPoint use the deck's palette.
async function applyTheme(file, theme) {
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const part = "ppt/theme/theme1.xml";
  const slots = ["dk1", "lt1", "dk2", "lt2", "accent1", "accent2", "accent3", "accent4", "accent5", "accent6", "hlink", "folHlink"];
  const scheme = `<a:clrScheme name="${theme.name}">` +
    slots.map((k) => `<a:${k}><a:srgbClr val="${theme.colors[k]}"/></a:${k}>`).join("") + "</a:clrScheme>";
  const xml = (await zip.file(part).async("string"))
    .replace(/<a:clrScheme\b[\s\S]*?<\/a:clrScheme>/, () => scheme)
    .replace(/(<a:(?:theme|fontScheme)\b[^>]*?\bname=")[^"]*"/g, (_, head) => `${head}${theme.name}"`);
  if (!xml.includes(scheme)) throw new Error(`${part} has no colour scheme to replace`);
  zip.file(part, xml);
  for (const name of Object.keys(zip.files).filter((n) => n.endsWith(".xml"))) {
    const bad = (await zip.file(name).async("string")).match(/<a:srgbClr val="((?![0-9A-Fa-f]{6}")[^"]*)"/);
    if (bad) throw new Error(`${name}: colour "${bad[1]}" is not six hex digits (scheme colour in a hex-only option)`);
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

const THEME = {
  name: "Deepfake Forensics",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0F172A", lt1: "FFFFFF", dk2: "1E1B4B", lt2: "EEF2FF",
    accent1: "4F46E5", accent2: "DC2626", accent3: "16A34A", accent4: "D97706",
    accent5: "64748B", accent6: "0EA5E9", hlink: "4F46E5", folHlink: "6D28D9",
  },
};
const H = THEME.colors; // hex, only for hex-only options (chart colors, grid lines, backgrounds)

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Robust Multimodal Deepfake Analysis";
pres.subject = "Project review";
const C = pres.SchemeColor;
const W = 13.333;
const MX = 0.6;
const CW = W - 2 * MX;

// ---------------------------------------------------------------- layouts
const footer = [
  { text: { text: "Robust Multimodal Deepfake Analysis", options: { x: MX, y: 7.0, w: 6, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
];
pres.defineSlideMaster({
  title: "TITLE", background: { color: H.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 2.2, w: CW, h: 1.5, fontSize: 48, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: MX, y: 3.85, w: 10.5, h: 0.9, fontSize: 20, color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: H.lt1 },
  objects: [
    ...footer,
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.35, w: CW, h: 0.85, fontSize: 32, bold: true, color: C.text1, valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.55, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});
pres.defineSlideMaster({
  title: "CONTENT_DARK", background: { color: H.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.35, w: CW, h: 0.85, fontSize: 32, bold: true, color: C.background1, valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.55, h: 0.3, fontSize: 10, color: C.background2, align: "right" },
});

// ---------------------------------------------------------------- helpers
async function iconPng(Icon, hex) {
  const svg = renderToStaticMarkup(React.createElement(Icon, { color: "#" + hex, size: 256 }));
  return "image/png;base64," + (await sharp(Buffer.from(svg)).png().toBuffer()).toString("base64");
}
const icons = {};
async function loadIcons() {
  const names = { face: fa.FaUserSecret, mic: fa.FaMicrophoneAlt, wave: fa.FaWaveSquare, shield: fa.FaShieldAlt,
    eye: fa.FaEye, layers: fa.FaLayerGroup, flask: fa.FaFlask, laptop: fa.FaLaptopCode, chart: fa.FaChartLine,
    code: fa.FaCode, info: fa.FaInfoCircle, film: fa.FaFilm, compress: fa.FaCompressArrowsAlt, check: fa.FaCheck,
    clock: fa.FaHourglassHalf, db: fa.FaDatabase, cogs: fa.FaCogs, balance: fa.FaBalanceScale };
  for (const [k, Icon] of Object.entries(names)) icons[k] = await iconPng(Icon, "FFFFFF");
}
function iconCircle(slide, key, x, y, d, fill, name) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill || C.accent1 }, line: { type: "none" }, objectName: name + " circle" });
  const p = d * 0.26;
  slide.addImage({ data: icons[key], x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, objectName: name + " icon" });
}
function card(slide, x, y, w, h, name, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.12, fill: { color: fill || C.background2 }, line: { type: "none" }, objectName: name,
  });
}
function pill(slide, text, x, y, w, kind, name) {
  const fill = { done: C.accent3, pending: C.accent4, info: C.accent1 }[kind];
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.32, rectRadius: 0.16, fill: { color: fill }, line: { type: "none" }, objectName: name + " pill" });
  slide.addText(text, { x, y, w, h: 0.32, fontSize: 11, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: name + " pill text" });
}
function text(slide, t, o) { slide.addText(t, { isTextBox: true, margin: 0, valign: "top", color: C.text1, fontSize: 15, ...o }); }
function bullets(items, size = 15) {
  return items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6, fontSize: size } }));
}
const chartText = (o) => ({
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
  titleFontFace: "+mn-lt", catAxisTitleFontFace: "+mn-lt", valAxisTitleFontFace: "+mn-lt",
  catAxisLabelColor: H.accent5, valAxisLabelColor: H.accent5, catAxisLabelFontSize: 12, valAxisLabelFontSize: 12,
  titleColor: H.dk1, titleFontSize: 14, legendFontSize: 12, legendColor: H.dk1,
  valGridLine: { color: "E2E8F0", size: 0.75 }, catGridLine: { style: "none" },
  catAxisLineShow: false, valAxisLineShow: false, ...o,
});
const m = D.metrics;
const pct = (v, d = 1) => (100 * v).toFixed(d) + "%";

async function build() {
  await loadIcons();

  // 1 ---------------------------------------------------------------- title
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Introduction" });
  s.addText("Robust Multimodal Deepfake Analysis", { placeholder: "title" });
  s.addText("Detecting fake faces and cloned voices, robust to real-world degradation and explained with Grad-CAM", { placeholder: "body" });
  const chips = [["face", "Face model"], ["mic", "Voice model"], ["layers", "Fusion"], ["eye", "Grad-CAM"]];
  chips.forEach(([k, label], i) => {
    const x = MX + i * 2.35;
    iconCircle(s, k, x, 5.25, 0.55, C.accent1, "title " + label);
    text(s, label, { x: x + 0.7, y: 5.25, w: 1.5, h: 0.55, valign: "middle", fontSize: 15, color: C.background1 });
  });
  text(s, "Project review · October 2026", { x: MX, y: 6.45, w: 6, h: 0.4, fontSize: 14, color: C.background2 });
  s.addNotes("Title. The project detects deepfake videos by checking both the face and the voice, tests how robust the detector is to compression and noise, and explains every decision with heatmaps. Today: what is built, the first real trained model (the voice model), the demo, and what is still pending.");

  // 2 ---------------------------------------------------------------- problem
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("Fakes can hit the face, the voice, or both", { placeholder: "title" });
  const probs = [
    ["face", "Fake faces", "Face swap, face reenactment and lip-sync models produce realistic manipulated faces."],
    ["mic", "Cloned voices", "Text-to-speech and voice conversion imitate a real speaker's voice convincingly."],
    ["compress", "Real-world damage", "Compression, noise and missing audio tracks break detectors that check only one channel."],
  ];
  probs.forEach(([k, h, b], i) => {
    const x = MX + i * (3.9 + 0.2);
    card(s, x, 1.55, 3.9, 3.55, "problem card " + (i + 1));
    iconCircle(s, k, x + 0.35, 1.85, 0.8, C.accent1, "problem " + h);
    text(s, h, { x: x + 0.35, y: 2.9, w: 3.2, h: 0.5, fontSize: 21, bold: true });
    text(s, b, { x: x + 0.35, y: 3.45, w: 3.2, h: 1.5, fontSize: 15 });
  });
  text(s, "A practical detector must answer “is this video fake?” even when only one channel was manipulated, and show evidence a person can check.",
    { x: MX, y: 5.5, w: CW, h: 0.9, fontSize: 18, italic: true, color: C.accent1 });
  s.addNotes("Problem. Deepfakes can manipulate the face, the voice, or both. A face-only detector misses a real face with a cloned voice, and an audio-only detector misses a face swap with the original audio. Social platforms also re-compress videos, which is known to hurt detectors. So we need a multimodal, robust and explainable detector.");

  // 3 ---------------------------------------------------------------- objectives
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("Four objectives", { placeholder: "title" });
  const objs = [
    ["layers", "Detect and combine", "Score the face and the voice separately, then fuse them into one decision: fake if either is fake."],
    ["shield", "Stay robust", "Measure and improve accuracy under 7 degradations × 3 severities and missing modalities."],
    ["eye", "Explain decisions", "Grad-CAM heatmaps on face crops and on the voice spectrogram."],
    ["laptop", "Reproducible, with a demo", "Tested pipeline, Colab notebooks for training, and an interactive Gradio app."],
  ];
  objs.forEach(([k, h, b], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = MX + col * 6.2, y = 1.6 + row * 2.45;
    card(s, x, y, 5.9, 2.15, "objective card " + (i + 1));
    iconCircle(s, k, x + 0.35, y + 0.4, 0.85, C.accent1, "objective " + h);
    text(s, h, { x: x + 1.5, y: y + 0.38, w: 4.1, h: 0.5, fontSize: 21, bold: true });
    text(s, b, { x: x + 1.5, y: y + 0.92, w: 4.1, h: 1.1, fontSize: 15 });
  });
  s.addNotes("Objectives. One: detect fake faces and fake voices and combine them. Two: stay reliable under compression, blur, noise and missing audio or video. Three: explain each decision with heatmaps. Four: deliver a reproducible pipeline and a working demo.");

  // 4 ---------------------------------------------------------------- literature
  pres.addSection({ title: "Method" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Related work and the gaps we address", { placeholder: "title" });
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 } } });
  const lit = [
    ["FaceForensics++ (Rössler et al., 2019)", "Benchmark of 4 face manipulations at several compression levels", "Accuracy drops on compressed video → robustness tests"],
    ["Celeb-DF (Li et al., 2020)", "High-quality face swaps for cross-dataset evaluation", "Generalisation → cross-dataset test of the face model"],
    ["ASVspoof 2019 LA (Wang et al., 2020)", "Spoofed speech from many TTS / voice-conversion systems; unseen attacks in evaluation", "Audio only → combined with the face"],
    ["AASIST (Jung et al., 2022)", "Graph-attention anti-spoofing countermeasure", "Stronger audio model → future work"],
    ["FakeAVCeleb (Khalid et al., 2021)", "Audio-visual fakes with separate face and voice labels", "Shows which modality a detector relies on"],
    ["Grad-CAM (Selvaraju et al., 2017)", "Class-discriminative heatmaps from CNN gradients", "Evidence for both branches"],
  ];
  s.addTable([[hdr("Work"), hdr("Contribution"), hdr("Gap this project addresses")],
    ...lit.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: j === 0, fill: { color: i % 2 ? C.background1 : C.background2 } } })))],
  { x: MX, y: 1.5, w: CW, colW: [3.5, 4.75, 3.88], fontSize: 14, color: C.text1, valign: "middle",
    border: { type: "solid", pt: 0.75, color: "CBD5E1" }, rowH: 0.7, margin: [4, 8, 4, 8], objectName: "literature table" });
  s.addNotes("Related work. FaceForensics++ showed that compression hurts detectors, which motivates the robustness tests. Celeb-DF is our cross-dataset face test. ASVspoof 2019 is the standard voice anti-spoofing dataset and the one we trained on. FakeAVCeleb labels face and voice separately, so we can tell which modality a detector relies on. Grad-CAM gives the explanations.");

  // 5 ---------------------------------------------------------------- datasets
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Datasets and leakage-free splits", { placeholder: "title" });
  const ds = [
    ["FakeAVCeleb v1.2", "Main face + voice training and test", "Awaiting access"],
    ["DFDC sample", "Pipeline development, face baseline", "Pending"],
    ["ASVspoof 2019 LA", "Voice training, per-attack evaluation", "Used"],
    ["Celeb-DF v2", "Cross-dataset face test", "Awaiting access"],
  ];
  s.addTable([[hdr("Dataset"), hdr("Role"), hdr("Status")],
    ...ds.map((r, i) => r.map((c, j) => ({ text: c, options: {
      bold: j !== 1, color: j === 2 ? (c === "Used" ? C.accent3 : C.accent4) : C.text1,
      fill: { color: i % 2 ? C.background1 : C.background2 } } })))],
  { x: MX, y: 1.55, w: 6.3, colW: [2.1, 2.7, 1.5], fontSize: 14, valign: "middle", rowH: 0.62,
    border: { type: "solid", pt: 0.75, color: "CBD5E1" }, margin: [4, 8, 4, 8], objectName: "dataset table" });
  text(s, "Splits group by person ID (FakeAVCeleb), source video (DFDC) or official protocol (ASVspoof, Celeb-DF test list), so the same face or voice never appears in both training and test. Training refuses data whose groups or files cross splits.",
    { x: MX, y: 4.95, w: 6.3, h: 1.6, fontSize: 14, color: C.accent5 });
  text(s, "ASVspoof subset used for the voice model", { x: 7.35, y: 1.55, w: 5.4, h: 0.4, fontSize: 18, bold: true });
  s.addTable([[hdr("Split"), hdr("Official partition"), hdr("Clips"), hdr("Spoof"), hdr("Bona fide")],
    ...[["Train", "train"], ["Validation", "development"], ["Test", "evaluation"]].map((r, i) =>
      [...r, "5,000", "2,500", "2,500"].map((c) => ({ text: c, options: { fill: { color: i % 2 ? C.background1 : C.background2 } } })))],
  { x: 7.35, y: 2.05, w: 5.38, colW: [1.15, 1.43, 0.9, 0.9, 1.0], fontSize: 14, valign: "middle", rowH: 0.55, color: C.text1,
    border: { type: "solid", pt: 0.75, color: "CBD5E1" }, margin: [4, 6, 4, 6], objectName: "asvspoof split table" });
  text(s, "Balanced subset: up to 2,500 clips per class from each official partition (seed 42). Evaluation attacks A07–A19 are mostly absent from training, so the test measures generalisation to new attacks.",
    { x: 7.35, y: 4.4, w: 5.38, h: 1.4, fontSize: 14, color: C.accent5 });
  s.addNotes("Datasets. FakeAVCeleb is the main dataset but needs author approval, which is still pending, as is Celeb-DF. ASVspoof 2019 LA is free and is what we trained the voice model on: a balanced subset of 5,000 clips per split taken from the official partitions. Splits are leakage-free: the same person or voice never appears in both training and test.");

  // 6 ---------------------------------------------------------------- architecture
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Architecture: two branches, one decision", { placeholder: "title" });
  const box = (t, x, y, w, h, fill, color, name, size = 14) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: fill }, line: { type: "none" }, objectName: name });
    text(s, t, { x: x + 0.08, y, w: w - 0.16, h, fontSize: size, bold: true, color, align: "center", valign: "middle" });
  };
  const arrow = (x1, y1, x2, y2, name) => s.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001,
    flipV: y2 < y1, line: { color: C.accent5, width: 1.75, endArrowType: "triangle" }, objectName: name });
  const bw = 1.75, bh = 0.95, gap = 0.35;
  const colX = (i) => MX + i * (bw + gap);
  const yV = 1.75, yA = 3.55;
  [["Video", 0], ["16 frames, uniform", 1], ["YuNet face crops 224²", 2], ["EfficientNet-B0 face model", 3]].forEach(([t, i]) =>
    box(t, colX(i), yV, bw, bh, i === 3 ? C.accent1 : C.background2, i === 3 ? C.background1 : C.text1, "video step " + i));
  [["Audio track", 0], ["16 kHz log-mel, 4 s window", 1], ["ResNet-18 voice model", 3]].forEach(([t, i]) =>
    box(t, colX(i), yA, bw, bh, i === 3 ? C.accent1 : C.background2, i === 3 ? C.background1 : C.text1, "audio step " + i));
  for (let i = 0; i < 3; i++) arrow(colX(i) + bw, yV + bh / 2, colX(i + 1), yV + bh / 2, "video arrow " + i);
  arrow(colX(0) + bw, yA + bh / 2, colX(1), yA + bh / 2, "audio arrow 0");
  arrow(colX(1) + bw, yA + bh / 2, colX(3), yA + bh / 2, "audio arrow 1");
  pill(s, "pending data", colX(3) + (bw - 1.4) / 2, yV - 0.45, 1.4, "pending", "face status");
  pill(s, "trained", colX(3) + (bw - 1.1) / 2, yA + bh + 0.13, 1.1, "done", "voice status");
  const fx = colX(4), fy = yV, fh = yA + bh - yV;
  const fw = 1.95;
  box("Fusion\nlate: max · noisy-OR · weighted\nintermediate: MLP + modality dropout", fx, fy, fw, fh, C.text2, C.background1, "fusion box", 13);
  arrow(colX(3) + bw, yV + bh / 2, fx, yV + bh / 2, "face to fusion");
  arrow(colX(3) + bw, yA + bh / 2, fx, yA + bh / 2, "voice to fusion");
  const ox = fx + fw + 0.35;
  box("P(fake)\n+ Grad-CAM evidence", ox, yV + 0.55, W - MX - ox, fh - 1.1, C.accent2, C.background1, "output box", 13);
  arrow(fx + fw, fy + fh / 2, ox, fy + fh / 2, "fusion to output");
  const labels = [["Face label", "video_label: a real face with a cloned voice counts as real"],
    ["Voice label", "audio_label: bona fide vs. spoofed speech"],
    ["Overall label", "fake if the face OR the voice is fake"]];
  labels.forEach(([h, b], i) => {
    const x = MX + i * 4.1;
    card(s, x, 5.25, 3.9, 1.3, "label card " + i);
    text(s, h, { x: x + 0.25, y: 5.4, w: 3.4, h: 0.35, fontSize: 15, bold: true });
    text(s, b, { x: x + 0.25, y: 5.78, w: 3.4, h: 0.7, fontSize: 14, color: C.accent5 });
  });
  s.addNotes("Architecture. The video branch samples 16 frames, finds faces with YuNet and scores each crop with EfficientNet-B0. The voice branch converts the audio to a log-mel spectrogram and scores a 4-second window with ResNet-18. Fusion combines them: late fusion with max, noisy-OR or a weighted average, or an intermediate MLP with modality dropout so it still works when one input is missing. The voice branch is trained; the face branch is waiting for FakeAVCeleb.");

  // 7 ---------------------------------------------------------------- training setup
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Preprocessing, training and evaluation", { placeholder: "title" });
  const cols3 = [
    ["film", "Preprocessing", ["16 frames per video, YuNet faces, 224² crops (margin 1.3)", "Audio: 16 kHz mono, first 10 s, 128-bin log-mel (25 ms window, 10 ms hop)", "Processed data stored on Google Drive"]],
    ["cogs", "Training recipe", ["AdamW, cosine schedule, mixed precision on a Colab T4", "Backbone frozen in epoch 1; head learning rate 10×", "SpecAugment for audio; class-balanced loss", "Early stopping on validation AUC, resumable checkpoints"]],
    ["balance", "Evaluation protocol", ["Primary metric: AUC; also EER", "Accuracy, precision, recall, F1 at threshold 0.5", "Best checkpoint chosen on validation; test evaluated once", "Per-category results (per attack system)"]],
  ];
  cols3.forEach(([k, h, items], i) => {
    const x = MX + i * 4.1;
    card(s, x, 1.55, 3.9, 3.75, "setup card " + i);
    iconCircle(s, k, x + 0.3, 1.8, 0.7, C.accent1, "setup " + h);
    text(s, h, { x: x + 1.15, y: 1.8, w: 2.6, h: 0.7, fontSize: 20, bold: true, valign: "middle" });
    s.addText(bullets(items, 14), { x: x + 0.3, y: 2.75, w: 3.35, h: 2.45, isTextBox: true, margin: 0, valign: "top", color: C.text1 });
  });
  const run = [["Hardware", "Colab T4 GPU"], ["Time per epoch", "≈ 2.6 min"], ["Epochs run", "6 of max 15"], ["Batch · learning rate", "32 · 1e-4"]];
  run.forEach(([k, v], i) => {
    const x = MX + i * 3.06;
    text(s, v, { x, y: 5.6, w: 2.9, h: 0.6, fontSize: 26, bold: true, color: C.accent1 });
    text(s, k, { x, y: 6.2, w: 2.9, h: 0.35, fontSize: 13, color: C.accent5 });
  });
  s.addNotes("Setup. Preprocessing is identical in training and in the demo. Training uses AdamW with a cosine schedule on a free Colab T4, freezes the pretrained backbone for the first epoch, and stops early on validation AUC. The test split is evaluated once with the best validation checkpoint.");

  // 8 ---------------------------------------------------------------- voice results
  pres.addSection({ title: "Results" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Voice model: AUC 0.977 and EER 6.2% on the test set", { placeholder: "title" });
  const stats = [[m.auc.toFixed(3), "Test AUC", "ranking quality, 1.0 = perfect"],
    [pct(m.eer), "Equal error rate", "lower is better"],
    [pct(m.accuracy), "Accuracy at 0.5", "precision 0.991 · recall 0.832"],
    [pct(m.at_eer_threshold.accuracy), "Accuracy at EER threshold", "threshold " + m.eer_threshold.toFixed(4)]];
  stats.forEach(([v, l, sub], i) => {
    const x = MX + (i % 2) * 3.1, y = 1.5 + Math.floor(i / 2) * 1.7;
    card(s, x, y, 2.95, 1.55, "stat card " + i);
    text(s, v, { x: x + 0.25, y: y + 0.12, w: 2.5, h: 0.7, fontSize: 36, bold: true, color: i < 2 ? C.accent1 : C.text1 });
    text(s, l, { x: x + 0.25, y: y + 0.82, w: 2.5, h: 0.32, fontSize: 14, bold: true });
    text(s, sub, { x: x + 0.25, y: y + 1.14, w: 2.6, h: 0.3, fontSize: 11, color: C.accent5 });
  });
  const [[tn, fp], [fn, tp]] = m.confusion_matrix;
  const cm = (t, ok, bold) => ({ text: t, options: { bold, align: "center", color: C.text1, fill: { color: ok === undefined ? C.background1 : ok ? "DCFCE7" : "FEE2E2" } } });
  s.addTable([[cm("", undefined), cm("Predicted real", undefined, true), cm("Predicted fake", undefined, true)],
    [cm("Actually real", undefined, true), cm(tn.toLocaleString("en-US"), true, true), cm(String(fp), false, true)],
    [cm("Actually fake", undefined, true), cm(String(fn), false, true), cm(tp.toLocaleString("en-US"), true, true)]],
  { x: MX, y: 5.0, w: 6.05, colW: [1.85, 2.1, 2.1], fontSize: 14, rowH: 0.42, valign: "middle",
    border: { type: "solid", pt: 1, color: H.lt1 }, objectName: "confusion matrix" });
  text(s, "n = 5,000 test clips; threshold 0.5", { x: MX, y: 6.33, w: 6, h: 0.3, fontSize: 11, color: C.accent5 });
  const roc = D.roc;
  const xs = roc.map((p) => p[0]);
  s.addChart(pres.charts.SCATTER, [
    { name: "FPR", values: xs },
    { name: "Voice model (AUC 0.977)", values: roc.map((p) => p[1]) },
    { name: "Chance", values: xs },
  ], chartText({ x: 7.0, y: 1.45, w: 5.75, h: 5.2, showTitle: true, title: "ROC curve, test split",
    chartColors: [H.accent1, "94A3B8"], lineSize: 2.5, lineDataSymbol: "none",
    showLegend: true, legendPos: "b", valAxisMaxVal: 1, valAxisMinVal: 0, valAxisMajorUnit: 0.2, catAxisMaxVal: 1, catAxisMinVal: 0, catAxisMajorUnit: 0.2,
    showValAxisTitle: true, valAxisTitle: "True positive rate", valAxisTitleColor: H.accent5, valAxisTitleFontSize: 12,
    showCatAxisTitle: true, catAxisTitle: "False positive rate", catAxisTitleColor: H.accent5, catAxisTitleFontSize: 12,
    objectName: "roc chart" }));
  s.addNotes(`Main voice result. On the 5,000 test clips from ASVspoof's evaluation partition, which mostly contains attacks never seen in training, the voice model reaches AUC 0.977 and an equal error rate of 6.2%. At the default threshold of 0.5 it accepts almost every real voice, with ${fp} false alarms out of 2,500, but misses ${fn} of 2,500 fakes. At the equal-error threshold accuracy rises to 93.8%. The ROC curve is computed from all 5,000 saved test predictions.`);

  // 9 ---------------------------------------------------------------- per attack
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Most unseen attacks are caught; A17 slips through", { placeholder: "title" });
  const atk = D.cats.filter((c) => c.category !== "bonafide");
  const rate = atk.map((c) => +(100 * c.accuracy).toFixed(1));
  s.addChart(pres.charts.BAR, [
    { name: "Detected ≥ 90%", labels: atk.map((c) => c.category), values: rate.map((r) => (r >= 90 ? r : 0)) },
    { name: "Detected < 90%", labels: atk.map((c) => c.category), values: rate.map((r) => (r < 90 ? r : 0)) },
  ], chartText({ x: MX, y: 1.45, w: 7.9, h: 5.25, barDir: "col", barGrouping: "stacked", barGapWidthPct: 45,
    chartColors: [H.accent1, H.accent2], showTitle: true, title: "Spoofed clips detected at threshold 0.5, per attack system (%)",
    showValue: true, dataLabelPosition: "inEnd", dataLabelColor: "FFFFFF", dataLabelFontSize: 11, dataLabelFormatCode: "0;;;",
    valAxisMaxVal: 100, valAxisMinVal: 0, valAxisMajorUnit: 25, showLegend: true, legendPos: "b", objectName: "per attack chart" }));
  const facts = [
    ["check", C.accent3, "8 of 13 attack systems", "94% or more detected, AUC ≥ 0.997"],
    ["balance", C.accent4, "A10, A12, A13, A18", "AUC ≥ 0.985 but only 64–80% detected at 0.5: the threshold is the problem"],
    ["info", C.accent2, "A17: 1.2% detected", "AUC 0.725, EER 34.4%; scores almost like real speech"],
    ["mic", C.accent1, "Real speech: 99.2% accepted", "19 false alarms in 2,500 bona fide clips"],
  ];
  facts.forEach(([k, col, h, b], i) => {
    const y = 1.5 + i * 1.3;
    iconCircle(s, k, 8.85, y + 0.05, 0.6, col, "attack fact " + i);
    text(s, h, { x: 9.6, y, w: 3.15, h: 0.35, fontSize: 15, bold: true });
    text(s, b, { x: 9.6, y: y + 0.38, w: 3.15, h: 0.85, fontSize: 13, color: C.accent5 });
  });
  s.addNotes("Per attack. Eight of the thirteen evaluation attacks are detected 94% of the time or more. Four attacks, A10, A12, A13 and A18, are ranked above real speech with AUC 0.985 or higher, yet only 64 to 80% cross the 0.5 threshold, so a calibrated threshold would catch many more. A17 is the hard case: only 1.2% detected, AUC 0.725. A16 and A19 reuse training algorithms A04 and A06, which explains their near-perfect scores.");

  // 10 ---------------------------------------------------------------- training curves
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Training converged fast; validation then saturated", { placeholder: "title" });
  const ep = D.history.map((h) => String(h.epoch));
  s.addChart(pres.charts.LINE, [
    { name: "Train loss", labels: ep, values: D.history.map((h) => +h.train_loss.toFixed(4)) },
    { name: "Validation loss", labels: ep, values: D.history.map((h) => +h.val_loss.toFixed(4)) },
  ], chartText({ x: MX, y: 1.45, w: 5.95, h: 3.9, chartColors: [H.accent1, H.accent6], lineSize: 2.5, lineDataSymbolSize: 7,
    showTitle: true, title: "Loss per epoch", showLegend: true, legendPos: "b", showCatAxisTitle: true, catAxisTitle: "Epoch",
    catAxisTitleColor: H.accent5, catAxisTitleFontSize: 12, objectName: "loss chart" }));
  s.addChart(pres.charts.LINE, [
    { name: "Validation AUC", labels: ep, values: D.history.map((h) => +h.val_auc.toFixed(4)) },
    { name: "Validation accuracy", labels: ep, values: D.history.map((h) => +h.val_acc.toFixed(4)) },
  ], chartText({ x: 6.78, y: 1.45, w: 5.95, h: 3.9, chartColors: [H.accent1, H.accent3], lineSize: 2.5, lineDataSymbolSize: 7,
    showTitle: true, title: "Validation metrics per epoch", showLegend: true, legendPos: "b", valAxisMinVal: 0.6, valAxisMaxVal: 1.0,
    valAxisMajorUnit: 0.1, showCatAxisTitle: true, catAxisTitle: "Epoch", catAxisTitleColor: H.accent5, catAxisTitleFontSize: 12,
    objectName: "validation chart" }));
  card(s, MX, 5.55, CW, 1.15, "training note card");
  text(s, [
    { text: "Why it stopped at epoch 6: ", options: { bold: true } },
    { text: "the validation partition contains the same attacks as training (A01–A06), so validation AUC reached 1.0 by epoch 3 and could not improve. Early stopping kept epoch 3. The test AUC of 0.977 on mostly unseen attacks is the meaningful measure." },
  ], { x: MX + 0.3, y: 5.65, w: CW - 0.6, h: 0.95, fontSize: 14, valign: "middle" });
  s.addNotes("Training. Epoch 1 trains only the new head with the backbone frozen, which is why validation AUC starts at 0.85. From epoch 2 the model separates the validation set almost perfectly, because the development partition shares its attack systems with training. Early stopping triggered after three epochs without improvement, and the epoch-3 checkpoint was kept. A validation set with held-out attacks would be a better guide; this is listed as future work.");

  // 11 ---------------------------------------------------------------- explainability
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Grad-CAM shows where the voice model looks", { placeholder: "title" });
  card(s, MX, 1.45, CW, 3.75, "gradcam frame", C.background1);
  s.addImage({ path: path.join(REPO, "reports/figures/gradcam_voice_example.png"), x: 0.95, y: 1.55, w: 11.43, h: 3.3, objectName: "gradcam spectrogram",
    altText: "Spectrogram of a synthetic voice with a Grad-CAM heatmap; the most influential region is in the first 1.8 seconds" });
  text(s, "Synthetic speech (macOS text-to-speech) analysed in the demo: P(fake) = 100%. Red marks the time–frequency regions that drove the decision.",
    { x: MX, y: 4.95, w: CW, h: 0.4, fontSize: 12, color: C.accent5 });
  const xp = [["wave", "Voice heatmaps", "Gradients of the decision at ResNet-18's last block (4 × 13 cells), upsampled onto the 128 × 401 spectrogram."],
    ["face", "Face heatmaps", "Per-frame maps on the face crops (EfficientNet-B0). Shown once the face model is trained on FakeAVCeleb."]];
  xp.forEach(([k, h, b], i) => {
    const x = MX + i * 6.2;
    iconCircle(s, k, x, 5.6, 0.65, i ? C.accent4 : C.accent1, "xai " + h);
    text(s, h, { x: x + 0.85, y: 5.55, w: 5.1, h: 0.35, fontSize: 16, bold: true });
    text(s, b, { x: x + 0.85, y: 5.92, w: 5.1, h: 0.8, fontSize: 13, color: C.accent5 });
  });
  s.addNotes("Explainability. This is the heatmap the demo produced for a synthetic voice made with macOS text-to-speech. The model flagged it as fake with 100% probability, and the red region shows which part of the spectrogram drove that decision. Heatmaps show where the model looked, not proof of manipulation. The same method works on face crops once the face model is trained.");

  // 12 ---------------------------------------------------------------- demo
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Demo app: verdict, evidence, model card, code", { placeholder: "title" });
  s.addImage({ path: path.join(REPO, "reports/figures/demo_app.png"), x: MX, y: 1.5, w: 7.6, h: 7.6 * 696 / 1190, objectName: "demo screenshot",
    altText: "Demo app showing a FAKE verdict at 100% with evidence per model branch" });
  text(s, "python app/app.py   ·   Colab: --share for a public link", { x: MX, y: 6.15, w: 7.6, h: 0.35, fontSize: 12, color: C.accent5, fontFace: "Courier New" });
  const tabs = [["film", "Analyse", "Video, audio or microphone; verdict gauge, score per branch, Grad-CAM"],
    ["chart", "Model & training", "Splits, settings, test metrics, per-attack table, curves, ROC"],
    ["code", "Code", "Exact training command and model source"],
    ["info", "About", "How it works, how to read a result, limitations"]];
  tabs.forEach(([k, h, b], i) => {
    const y = 1.5 + i * 1.22;
    iconCircle(s, k, 8.55, y + 0.05, 0.6, C.accent1, "tab " + h);
    text(s, h, { x: 9.3, y, w: 3.45, h: 0.35, fontSize: 16, bold: true });
    text(s, b, { x: 9.3, y: y + 0.37, w: 3.45, h: 0.75, fontSize: 13, color: C.accent5 });
  });
  s.addNotes("Demo. The Gradio app takes a video, an audio file or a microphone recording. It shows the verdict with a probability gauge and an adjustable threshold, the score of each branch, the input checks and the heatmap. The Model and training tab shows the data, settings and test results of the loaded model, and the Code tab shows the exact command that trained it. Live demo here.");

  // 13 ---------------------------------------------------------------- pending
  pres.addSection({ title: "Status and outlook" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Status and outlook" });
  s.addText("Built and tested, waiting for FakeAVCeleb", { placeholder: "title" });
  const pend = [
    ["face", "Face model", "EfficientNet-B0 on FakeAVCeleb and DFDC; cross-dataset test on Celeb-DF"],
    ["layers", "Fusion comparison", "Face-only vs voice-only vs late vs intermediate fusion, overall and per category"],
    ["shield", "Robustness", "7 degradations × 3 severities, missing modality, standard vs robust training"],
    ["mic", "Voice fine-tuning", "ASVspoof model fine-tuned on FakeAVCeleb; with vs without pre-training"],
  ];
  pend.forEach(([k, h, b], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = MX + col * 6.2, y = 1.55 + row * 2.2;
    card(s, x, y, 5.9, 1.95, "pending card " + i);
    iconCircle(s, k, x + 0.3, y + 0.35, 0.75, C.accent1, "pending " + h);
    text(s, h, { x: x + 1.3, y: y + 0.3, w: 2.8, h: 0.45, fontSize: 19, bold: true });
    pill(s, "pending data", x + 4.2, y + 0.36, 1.4, "pending", "pending " + i);
    text(s, b, { x: x + 1.3, y: y + 0.85, w: 4.35, h: 0.95, fontSize: 14 });
  });
  card(s, MX, 6.0, CW, 0.7, "tests card", C.background1);
  iconCircle(s, "check", MX, 6.05, 0.55, C.accent3, "tests");
  text(s, "Every code path above runs end to end on synthetic data: 82 automated tests pass. Only the real-data runs are missing.",
    { x: MX + 0.75, y: 6.05, w: CW - 0.75, h: 0.55, fontSize: 15, valign: "middle" });
  s.addNotes("Pending. Everything on this slide is implemented and covered by tests, but needs the FakeAVCeleb dataset, which requires the authors' approval. Once it arrives, the notebooks run the face model, the fusion comparison, the robustness study and the fine-tuning of the voice model.");

  // 14 ---------------------------------------------------------------- phases
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Status and outlook" });
  s.addText("Development status by phase", { placeholder: "title" });
  const phases = [["Setup", "done", "–"], ["Data pipeline", "done", "partial"], ["Face branch", "done", "pending"],
    ["Voice branch", "done", "done"], ["Fusion", "done", "pending"], ["Robustness", "done", "pending"],
    ["Explain-\nability", "done", "partial"], ["Demo app", "done", "done"], ["Report", "done", "partial"]];
  const pw = CW / phases.length;
  s.addShape(pres.shapes.LINE, { x: MX + pw / 2, y: 2.0, w: CW - pw, h: 0, line: { color: "CBD5E1", width: 2 }, objectName: "timeline line" });
  const stCol = { done: C.accent3, partial: C.accent1, pending: C.accent4, "–": C.accent5 };
  const stTxt = { done: "done", partial: "partial", pending: "pending", "–": "n/a" };
  phases.forEach(([name, code, real], i) => {
    const cx = MX + i * pw + pw / 2;
    s.addShape(pres.shapes.OVAL, { x: cx - 0.33, y: 1.67, w: 0.66, h: 0.66, fill: { color: real === "pending" ? C.accent4 : C.accent1 }, line: { color: H.lt1, width: 3 }, objectName: "phase node " + i });
    text(s, String(i), { x: cx - 0.33, y: 1.67, w: 0.66, h: 0.66, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle" });
    text(s, name, { x: cx - pw / 2 + 0.03, y: 2.5, w: pw - 0.06, h: 0.75, fontSize: 14, bold: true, align: "center" });
    text(s, "Code", { x: cx - pw / 2, y: 3.4, w: pw, h: 0.3, fontSize: 11, color: C.accent5, align: "center" });
    text(s, stTxt[code], { x: cx - pw / 2, y: 3.7, w: pw, h: 0.35, fontSize: 14, bold: true, color: stCol[code], align: "center" });
    text(s, "Real data", { x: cx - pw / 2, y: 4.25, w: pw, h: 0.3, fontSize: 11, color: C.accent5, align: "center" });
    text(s, stTxt[real], { x: cx - pw / 2, y: 4.55, w: pw, h: 0.35, fontSize: 14, bold: true, color: stCol[real], align: "center" });
  });
  card(s, MX, 5.35, CW, 1.3, "phase note card");
  text(s, bullets(["Partial: ASVspoof is processed, FakeAVCeleb, DFDC and Celeb-DF are not; Grad-CAM shown on the voice only; report section 6.2 (voice) is written.",
    "The trained voice model, its test results and every prediction are published on GitHub (models/ and a GitHub Release)."], 14),
  { x: MX + 0.3, y: 5.45, w: CW - 0.6, h: 1.1, valign: "middle" });
  s.addNotes("Status. All nine phases are implemented and tested. On real data, the voice branch and the demo are complete; the data pipeline, explainability and the report are partially complete; the face branch, fusion and robustness are waiting for data. The trained model and its results are published in the GitHub repository.");

  // 15 ---------------------------------------------------------------- limitations
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Status and outlook" });
  s.addText("Limitations and next steps", { placeholder: "title" });
  const lim = ["Subset results (5,000 clips per split), not the full ASVspoof benchmark",
    "Studio speech only: other microphones, rooms and compression are untested",
    "Threshold 0.5 under-detects several unseen attacks",
    "Validation shares attacks with training, so it saturates",
    "Face and fusion models not yet trained; Grad-CAM is coarse (4 × 13 on audio)"];
  const nxt = ["Obtain FakeAVCeleb: train face and fusion models, run robustness",
    "Validation with held-out attacks and a calibrated threshold",
    "Stronger audio models (AASIST, self-supervised speech features)",
    "Temporal and lip–audio synchronisation features",
    "Test on real recordings from everyday devices"];
  [["info", C.accent2, "Limitations", lim], ["chart", C.accent3, "Next steps", nxt]].forEach(([k, col, h, items], i) => {
    const x = MX + i * 6.2;
    card(s, x, 1.55, 5.9, 4.45, "limits card " + i);
    iconCircle(s, k, x + 0.3, 1.8, 0.7, col, h);
    text(s, h, { x: x + 1.2, y: 1.8, w: 4.4, h: 0.7, fontSize: 21, bold: true, valign: "middle" });
    s.addText(bullets(items, 16), { x: x + 0.3, y: 2.75, w: 5.3, h: 3.1, isTextBox: true, margin: 0, valign: "top", color: C.text1 });
  });
  s.addNotes("Limitations: these are subset results; the voice model has only seen studio-quality speech; the default threshold misses several unseen attacks; and the validation set cannot separate checkpoints because it shares attacks with training. Next steps: FakeAVCeleb for the face and fusion models, a held-out-attack validation set with a calibrated threshold, stronger audio models such as AASIST, and tests on everyday recordings.");

  // 16 ---------------------------------------------------------------- conclusion
  s = pres.addSlide({ masterName: "CONTENT_DARK", sectionTitle: "Status and outlook" });
  s.addText("Conclusion", { placeholder: "title" });
  const take = [["Working pipeline", "Data → training → evaluation → explainable demo, reproducible on Colab, 82 automated tests."],
    ["Voice model: AUC 0.977, EER 6.2%", "On mostly unseen ASVspoof attacks; strong on most, A17 remains the hard case."],
    ["Next: the multimodal claim", "FakeAVCeleb unlocks the face and fusion models, the core comparison still to be measured."]];
  take.forEach(([h, b], i) => {
    const y = 1.65 + i * 1.65;
    s.addShape(pres.shapes.OVAL, { x: MX, y, w: 0.8, h: 0.8, fill: { color: C.accent1 }, line: { type: "none" }, objectName: "takeaway " + i });
    text(s, String(i + 1), { x: MX, y, w: 0.8, h: 0.8, fontSize: 26, bold: true, color: C.background1, align: "center", valign: "middle" });
    text(s, h, { x: MX + 1.1, y: y - 0.05, w: 10.9, h: 0.5, fontSize: 24, bold: true, color: C.background1 });
    text(s, b, { x: MX + 1.1, y: y + 0.48, w: 10.9, h: 0.6, fontSize: 17, color: C.background2 });
  });
  text(s, "github.com/tanishq1101/robust-multimodal-deepfake", { x: MX, y: 6.6, w: 8, h: 0.4, fontSize: 14, color: C.background2 });
  s.addNotes("Conclusion. One: the full pipeline works end to end and is tested. Two: the first trained model, the voice model, reaches AUC 0.977 and EER 6.2% on mostly unseen attacks, with A17 as the hard case. Three: the core multimodal comparison needs FakeAVCeleb, which is the next step.");

  // 17 ---------------------------------------------------------------- references
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Status and outlook" });
  s.addText("References", { placeholder: "title" });
  const refs = [
    "Rössler et al. FaceForensics++. ICCV 2019.", "Li et al. Celeb-DF. CVPR 2020.",
    "Dolhansky et al. The DeepFake Detection Challenge Dataset. arXiv:2006.07397, 2020.",
    "Khalid et al. FakeAVCeleb. NeurIPS Datasets and Benchmarks, 2021.",
    "Wang et al. ASVspoof 2019. Computer Speech & Language 64, 2020.",
    "Jung et al. AASIST. ICASSP 2022.", "Tan & Le. EfficientNet. ICML 2019.",
    "He et al. Deep Residual Learning (ResNet). CVPR 2016.", "Park et al. SpecAugment. Interspeech 2019.",
    "Selvaraju et al. Grad-CAM. ICCV 2017.", "Neverova et al. ModDrop. IEEE TPAMI 38(8), 2016.",
    "Wu et al. YuNet. Machine Intelligence Research 20, 2023.",
  ];
  [refs.slice(0, 6), refs.slice(6)].forEach((col, i) => {
    s.addText(col.map((t, j) => ({ text: t, options: { breakLine: j < col.length - 1, paraSpaceAfter: 18, fontSize: 16 } })),
      { x: MX + i * 6.2, y: 1.55, w: 5.9, h: 5.0, isTextBox: true, margin: 0, valign: "top", color: C.text1 });
  });
  s.addNotes("Main references; the full list is in reports/report.md.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", path.relative(process.cwd(), OUT) || OUT);
}
build().catch((e) => { console.error(e); process.exit(1); });
