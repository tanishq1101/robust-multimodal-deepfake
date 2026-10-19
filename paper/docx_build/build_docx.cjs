// Build an IEEE-style two-column Word document from blocks.json (made by tex_to_json.py).
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType,
  BorderStyle, ShadingType, SectionType, TabStopType, VerticalAlign,
} = require('docx');

const here = __dirname;
const data = JSON.parse(fs.readFileSync(path.join(here, 'blocks.json'), 'utf8'));
const FONT = 'Times New Roman';
const BODY = 20;      // 10 pt
const SMALL = 16;     // 8 pt
const COLW = 5040;    // 3.5 in column

const mkRuns = (runs, base = {}) => runs.map(r => {
  const o = { text: r.t, font: r.code ? 'Courier New' : FONT, size: base.size || BODY };
  if (r.b || base.bold) o.bold = true;
  if (r.i || base.italics) o.italics = true;
  if (r.sub) o.subScript = true;
  if (r.sup) o.superScript = true;
  if (r.upright && !r.i && !r.sub && !r.sup) o.italics = base.italics || false;
  if (base.smallCaps) o.smallCaps = true;
  return new TextRun(o);
});
const tr = (text, o = {}) => new TextRun({ text, font: FONT, size: BODY, ...o });
const body = (runs, extra = {}) => new Paragraph({
  children: mkRuns(runs), alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 228 },
  indent: { firstLine: 202 }, ...extra,
});

const img = (b, extra = {}) => new ImageRun({
  type: 'png', data: fs.readFileSync(b.path), transformation: { width: Math.round(b.w), height: Math.round(b.h) }, ...extra,
});

function tableBlock(b) {
  const n = b.aligns.length;
  const lens = new Array(n).fill(4);
  b.rows.forEach(r => r.cells.forEach((c, i) => {
    const len = c.map(x => x.t).join('').length;
    lens[i] = Math.max(lens[i], Math.min(len, 38));
  }));
  const tot = lens.reduce((a, c) => a + c, 0);
  let widths = lens.map(l => Math.floor(COLW * l / tot));
  widths[n - 1] += COLW - widths.reduce((a, c) => a + c, 0);
  const none = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };
  const line = (sz) => ({ style: BorderStyle.SINGLE, size: sz, color: '000000' });
  const al = { l: AlignmentType.LEFT, r: AlignmentType.RIGHT, c: AlignmentType.CENTER };
  const rows = b.rows.map((r, ri) => new TableRow({
    cantSplit: true,
    children: r.cells.map((c, ci) => new TableCell({
      width: { size: widths[ci], type: WidthType.DXA },
      verticalAlign: VerticalAlign.CENTER,
      margins: { top: 15, bottom: 15, left: 50, right: 50 },
      borders: {
        top: ri === 0 ? line(8) : (r.rule_before ? line(4) : none),
        bottom: r.rule_after ? line(8) : none, left: none, right: none,
      },
      children: [new Paragraph({
        alignment: ri === 0 ? AlignmentType.CENTER === al[b.aligns[ci]] ? al[b.aligns[ci]] : al[b.aligns[ci]] : al[b.aligns[ci]],
        spacing: { after: 0, line: 200 },
        children: mkRuns(c, { size: SMALL }),
      })],
    })),
  }));
  const out = [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 0 },
      children: [tr(`TABLE ${b.num}`, { size: SMALL, smallCaps: true })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, keepNext: true,
      children: mkRuns(b.caption, { size: SMALL, smallCaps: true }) }),
    new Table({ width: { size: COLW, type: WidthType.DXA }, columnWidths: widths, rows }),
  ];
  if (b.note) out.push(new Paragraph({ spacing: { before: 30, after: 80 }, alignment: AlignmentType.JUSTIFIED,
    children: mkRuns(b.note, { size: 14 }) }));
  else out.push(new Paragraph({ spacing: { after: 80 }, children: [] }));
  return out;
}

function figureBlock(b) {
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 80, after: 40 }, children: [img(b)] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 },
      children: [tr(`Fig. ${b.num}. `, { size: SMALL }), ...mkRuns(b.caption, { size: SMALL })] }),
  ];
}

function eqBlock(b) {
  return new Paragraph({
    tabStops: [{ type: TabStopType.CENTER, position: COLW / 2 }, { type: TabStopType.RIGHT, position: COLW }],
    spacing: { before: 40, after: 40 },
    children: [tr('\t'), img(b), tr(`\t(${b.num})`)],
  });
}

// ---------------------------------------------------------------- assemble sections
const sections = [];
let cur = [];
let curType = 'two';
const pageProps = {
  page: { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1440, left: 900, right: 900 } },
};
function flush() {
  if (!cur.length) return;
  sections.push({
    properties: {
      ...pageProps,
      type: sections.length ? SectionType.CONTINUOUS : undefined,
      column: curType === 'two' ? { count: 2, space: 360, equalWidth: true } : { count: 1 },
    },
    children: cur,
  });
  cur = [];
}

// title block (single column)
curType = 'one';
cur.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
  children: [tr(data.title, { size: 48 })] }));
cur.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 },
  children: [tr('Author One', { size: 22 }), tr('1', { size: 22, superScript: true }), tr(', Author Two', { size: 22 }), tr('1', { size: 22, superScript: true })] }));
cur.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 },
  children: [tr('1', { size: 18, superScript: true }), tr('Department of Computer Science and Engineering, University Name, City, Country', { size: 18, italics: true })] }));
cur.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
  children: [tr('Email: author.one@university.edu, author.two@university.edu', { size: 18 })] }));
flush();
curType = 'two';

for (const b of data.blocks) {
  if (b.type === 'figure' && b.wide) {
    flush();
    curType = 'one';
    cur.push(...figureBlock(b));
    flush();
    curType = 'two';
    continue;
  }
  switch (b.type) {
    case 'abstract':
      cur.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 216 },
        children: [tr('Abstract—', { bold: true, italics: true, size: 18 }), ...mkRuns(b.runs, { size: 18, bold: true })] }));
      break;
    case 'keywords':
      cur.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 216 },
        children: [tr('Index Terms—', { bold: true, italics: true, size: 18 }), ...mkRuns(b.runs, { size: 18, bold: true, italics: true })] }));
      break;
    case 'h1':
      cur.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 80 },
        children: [tr(b.text, { size: BODY, smallCaps: false })] }));
      break;
    case 'h2':
      cur.push(new Paragraph({ keepNext: true, spacing: { before: 100, after: 40 },
        children: [tr(b.text, { italics: true })] }));
      break;
    case 'para':
      cur.push(body(b.runs));
      break;
    case 'enum':
      b.items.forEach((it, i) => cur.push(new Paragraph({
        alignment: AlignmentType.JUSTIFIED, spacing: { after: 40, line: 228 },
        indent: { left: 300, hanging: 300 },
        children: [tr(`${i + 1})\t`), ...mkRuns(it)],
        tabStops: [{ type: TabStopType.LEFT, position: 300 }],
      })));
      break;
    case 'eq':
      cur.push(eqBlock(b));
      break;
    case 'figure':
      cur.push(...figureBlock(b));
      break;
    case 'table':
      cur.push(...tableBlock(b));
      break;
  }
}
cur.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 80 },
  children: [tr('REFERENCES')] }));
data.refs.forEach((r, i) => cur.push(new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 20, line: 200 },
  indent: { left: 360, hanging: 360 },
  tabStops: [{ type: TabStopType.LEFT, position: 360 }],
  children: [tr(`[${i + 1}]\t`, { size: SMALL }), ...mkRuns(r, { size: SMALL })],
})));
flush();

const doc = new Document({
  creator: 'Author One',
  title: data.title,
  styles: { default: { document: { run: { font: FONT, size: BODY } } } },
  sections,
});
Packer.toBuffer(doc).then(buf => {
  const out = path.join(here, '..', 'paper_word.docx');
  fs.writeFileSync(out, buf);
  console.log('wrote', out);
});
