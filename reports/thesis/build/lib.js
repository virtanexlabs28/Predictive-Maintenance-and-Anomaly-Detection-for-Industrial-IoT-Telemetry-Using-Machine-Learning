/*
 * Builds one thesis chapter as a Word document from a list of content blocks.
 *
 * A chapter file describes its content as plain data (headings, paragraphs,
 * tables, figures, equations). This library turns that into a .docx and takes
 * care of numbering: sections, figures, tables, equations and citations are
 * numbered automatically, so nothing has to be renumbered by hand.
 *
 * Inline markup inside text:
 *   **bold**   *italic*   {cite:key1,key2}   {ref:id}
 */

const fs = require("fs");
const path = require("path");
const docx = require("docx");

const {
  AlignmentType, BorderStyle, Document, Footer, HeadingLevel, ImageRun, LevelFormat, LineRuleType,
  Packer, PageNumber, Paragraph, Table, TableCell, TableRow, TextRun, VerticalAlign, WidthType,
} = docx;

const ROOT = path.resolve(__dirname, "..", "..", "..");
const FIGURE_DIR = path.join(ROOT, "results", "figures");
const OUT_DIR = path.resolve(__dirname, "..");
const REFERENCES = JSON.parse(fs.readFileSync(path.join(__dirname, "references.json"), "utf8"));

const FONT = "Times New Roman";
const BODY = 24;            // 12 pt (sizes are in half-points)
const SMALL = 20;           // 10 pt, used in tables and captions
const TEXT_WIDTH = 9026;    // A4 with 2.54 cm margins, in twentieths of a point
const TEXT_WIDTH_CM = 15.92;
const LINE_15 = { line: 360, lineRule: LineRuleType.AUTO };
const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const NO_BORDERS = { top: NONE, bottom: NONE, left: NONE, right: NONE };

// ---------------------------------------------------------------- numbering

function collectStrings(block) {
  if (block.p) return [block.p];
  if (block.bullets) return block.bullets;
  if (block.numbered) return block.numbered;
  if (block.table) return [block.table.caption, ...block.table.header, ...block.table.rows.flat(), block.table.note || ""];
  if (block.figure) return [block.figure.caption];
  if (block.h) return [block.text];
  return [];
}

function buildContext(chapter, blocks) {
  const labels = {};
  const citations = [];
  const counters = { figure: 0, table: 0, equation: 0 };
  for (const block of blocks) {
    for (const kind of ["figure", "table", "equation"]) {
      if (!block[kind]) continue;
      counters[kind] += 1;
      const number = `${chapter}.${counters[kind]}`;
      block[kind].number = number;
      if (block[kind].id) {
        const name = { figure: `Figure ${number}`, table: `Table ${number}`, equation: `Equation (${number})` }[kind];
        if (labels[block[kind].id]) throw new Error(`Duplicate id: ${block[kind].id}`);
        labels[block[kind].id] = name;
      }
    }
    for (const text of collectStrings(block)) {
      for (const match of String(text).matchAll(/\{cite:([^}]+)\}/g)) {
        for (const key of match[1].split(",").map((k) => k.trim())) {
          if (!REFERENCES[key]) throw new Error(`Unknown reference key: ${key}`);
          if (!citations.includes(key)) citations.push(key);
        }
      }
    }
  }
  return {
    citations,
    cite: (keys) => keys.split(",").map((k) => citations.indexOf(k.trim()) + 1).sort((a, b) => a - b).map((n) => `[${n}]`).join(", "),
    ref: (id) => {
      if (!labels[id]) throw new Error(`Unknown cross-reference: ${id}`);
      return labels[id];
    },
  };
}

// ------------------------------------------------------------------- inline

function inline(text, ctx, base = {}) {
  const runs = [];
  const add = (value, extra = {}) => {
    if (value) runs.push(new TextRun({ text: value, font: FONT, size: BODY, ...base, ...extra }));
  };
  const pattern = /(\*\*[^*]+\*\*|\*[^*]+\*|\{cite:[^}]+\}|\{ref:[^}]+\})/g;
  let last = 0;
  for (const match of String(text).matchAll(pattern)) {
    add(text.slice(last, match.index));
    const token = match[0];
    if (token.startsWith("**")) add(token.slice(2, -2), { bold: true });
    else if (token.startsWith("*")) add(token.slice(1, -1), { italics: true });
    else if (token.startsWith("{cite:")) add(ctx.cite(token.slice(6, -1)));
    else add(ctx.ref(token.slice(5, -1)));
    last = match.index + token.length;
  }
  add(text.slice(last));
  return runs;
}

// ------------------------------------------------------------------- blocks

function bodyParagraph(text, ctx, extra = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { ...LINE_15, after: 160 },
    children: inline(text, ctx),
    ...extra,
  });
}

function listItems(items, ctx, reference, instance) {
  return items.map((item) => new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { ...LINE_15, after: 60 },
    numbering: { reference, level: 0, instance },
    children: inline(item, ctx),
  }));
}

function caption(label, text, ctx, { above }) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    keepNext: above,
    spacing: above ? { before: 200, after: 100 } : { before: 100, after: 240 },
    children: [
      new TextRun({ text: `${label}: `, font: FONT, size: SMALL + 2, bold: true }),
      ...inline(text, ctx, { size: SMALL + 2 }),
    ],
  });
}

function makeTable(spec, ctx) {
  const columns = spec.header.length;
  const weights = spec.widths || Array(columns).fill(1);
  const total = weights.reduce((a, b) => a + b, 0);
  const widths = weights.map((w) => Math.floor((TEXT_WIDTH * w) / total));
  widths[columns - 1] += TEXT_WIDTH - widths.reduce((a, b) => a + b, 0);
  const align = spec.align || Array(columns).fill("left");
  const thick = { style: BorderStyle.SINGLE, size: 12, color: "000000" };
  const thin = { style: BorderStyle.SINGLE, size: 6, color: "000000" };
  const size = spec.fontSize || SMALL;

  const makeRow = (cells, { header, last }) => new TableRow({
    tableHeader: header,
    cantSplit: true,
    children: cells.map((cell, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      verticalAlign: VerticalAlign.CENTER,
      margins: { top: 50, bottom: 50, left: 80, right: 80 },
      borders: {
        top: header ? thick : NONE,
        bottom: header ? thin : last ? thick : NONE,
        left: NONE,
        right: NONE,
      },
      children: [new Paragraph({
        alignment: { left: AlignmentType.LEFT, right: AlignmentType.RIGHT, center: AlignmentType.CENTER }[align[i]],
        spacing: { before: 0, after: 0 },
        children: inline(String(cell), ctx, { size, bold: header }),
      })],
    })),
  });

  const rows = [makeRow(spec.header, { header: true, last: false })];
  spec.rows.forEach((row, i) => rows.push(makeRow(row, { header: false, last: i === spec.rows.length - 1 })));
  const out = [
    caption(`Table ${spec.number}`, spec.caption, ctx, { above: true }),
    new Table({ width: { size: TEXT_WIDTH, type: WidthType.DXA }, columnWidths: widths, rows }),
  ];
  out.push(new Paragraph({
    spacing: { before: 60, after: 240 },
    children: spec.note ? inline(`*Note.* ${spec.note}`, ctx, { size: SMALL - 2 }) : [],
  }));
  return out;
}

function pngSize(file) {
  const buffer = fs.readFileSync(file);
  return { width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20), data: buffer };
}

function makeFigure(spec, ctx) {
  const file = path.join(FIGURE_DIR, spec.file);
  const { width, height, data } = pngSize(file);
  const widthCm = Math.min(spec.widthCm || TEXT_WIDTH_CM, TEXT_WIDTH_CM);
  const widthPx = Math.round((widthCm / 2.54) * 96);
  const heightPx = Math.round((widthPx * height) / width);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      keepNext: true,
      spacing: { before: 200, after: 0 },
      children: [new ImageRun({ type: "png", data, transformation: { width: widthPx, height: heightPx } })],
    }),
    caption(`Figure ${spec.number}`, spec.caption, ctx, { above: false }),
  ];
}

function makeEquation(spec) {
  // Borderless three-column table: empty | equation (centred) | number (right).
  const widths = [900, TEXT_WIDTH - 1800, 900];
  const cell = (children, i, alignment) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    borders: NO_BORDERS,
    children: [new Paragraph({ alignment, spacing: { before: 60, after: 60 }, children })],
  });
  return [
    new Table({
      width: { size: TEXT_WIDTH, type: WidthType.DXA },
      columnWidths: widths,
      borders: { ...NO_BORDERS, insideHorizontal: NONE, insideVertical: NONE },
      rows: [new TableRow({
        cantSplit: true,
        children: [
          cell([], 0, AlignmentType.LEFT),
          cell([new docx.Math({ children: spec.math })], 1, AlignmentType.CENTER),
          cell([new TextRun({ text: `(${spec.number})`, font: FONT, size: BODY })], 2, AlignmentType.RIGHT),
        ],
      })],
    }),
    new Paragraph({ spacing: { before: 0, after: 80 }, children: [] }),
  ];
}

// ----------------------------------------------------------------- document

async function buildChapter({ number, title, blocks, outFile }) {
  const ctx = buildContext(number, blocks);
  const children = [];
  let section = 0;
  let subsection = 0;
  let listInstance = 0;

  children.push(new Paragraph({
    heading: HeadingLevel.HEADING_1,
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: `Chapter ${number}: ${title}`, font: FONT, size: 32, bold: true })],
  }));

  for (const block of blocks) {
    if (block.h === 2) {
      section += 1;
      subsection = 0;
      children.push(new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun({ text: `${number}.${section} ${block.text}`, font: FONT, size: 28, bold: true })],
      }));
    } else if (block.h === 3) {
      subsection += 1;
      children.push(new Paragraph({
        heading: HeadingLevel.HEADING_3,
        children: [new TextRun({ text: `${number}.${section}.${subsection} ${block.text}`, font: FONT, size: BODY, bold: true })],
      }));
    } else if (block.p) {
      children.push(bodyParagraph(block.p, ctx));
    } else if (block.bullets) {
      listInstance += 1;
      children.push(...listItems(block.bullets, ctx, "bullets", listInstance));
      children.push(new Paragraph({ spacing: { before: 0, after: 100 }, children: [] }));
    } else if (block.numbered) {
      listInstance += 1;
      children.push(...listItems(block.numbered, ctx, "numbers", listInstance));
      children.push(new Paragraph({ spacing: { before: 0, after: 100 }, children: [] }));
    } else if (block.table) {
      children.push(...makeTable(block.table, ctx));
    } else if (block.figure) {
      children.push(...makeFigure(block.figure, ctx));
    } else if (block.equation) {
      children.push(...makeEquation(block.equation));
    } else {
      throw new Error(`Unknown block: ${JSON.stringify(block).slice(0, 80)}`);
    }
  }

  // Reference list for this chapter (numbers are unified when the thesis is assembled).
  if (ctx.citations.length > 0) {
    children.push(new Paragraph({
      heading: HeadingLevel.HEADING_2,
      children: [new TextRun({ text: `References cited in Chapter ${number}`, font: FONT, size: 28, bold: true })],
    }));
  }
  ctx.citations.forEach((key, i) => {
    children.push(new Paragraph({
      alignment: AlignmentType.LEFT,
      spacing: { line: 276, lineRule: LineRuleType.AUTO, after: 100 },
      indent: { left: 567, hanging: 567 },
      children: [new TextRun({ text: `[${i + 1}]\t`, font: FONT, size: BODY - 2 }), ...inline(REFERENCES[key].ieee, ctx, { size: BODY - 2 })],
    }));
  });

  const headingStyle = (size, before, after) => ({
    run: { font: FONT, size, bold: true, color: "000000" },
    paragraph: { spacing: { before, after }, keepNext: true },
  });
  const document = new Document({
    creator: "Thesis author",
    title: `Chapter ${number}: ${title}`,
    styles: {
      default: {
        document: { run: { font: FONT, size: BODY } },
        heading1: headingStyle(32, 0, 360),
        heading2: headingStyle(28, 360, 160),
        heading3: headingStyle(BODY, 240, 120),
      },
    },
    numbering: {
      config: [
        { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
        { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      ],
    },
    sections: [{
      properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 22 })],
          })],
        }),
      },
      children,
    }],
  });

  const target = path.join(OUT_DIR, outFile);
  fs.writeFileSync(target, await Packer.toBuffer(document));
  const words = blocks.flatMap(collectStrings).join(" ").split(/\s+/).filter(Boolean).length;
  console.log(`written ${path.relative(ROOT, target)} | ${ctx.citations.length} references | about ${words} words`);
  return target;
}

// Small helpers for writing equations in the chapter files.
const math = {
  t: (text) => new docx.MathRun(text),
  sub: (base, lower) => new docx.MathSubScript({ children: [new docx.MathRun(base)], subScript: [new docx.MathRun(lower)] }),
  sup: (base, upper) => new docx.MathSuperScript({ children: base, superScript: [new docx.MathRun(upper)] }),
  subsup: (base, lower, upper) => new docx.MathSubSuperScript({ children: [new docx.MathRun(base)], subScript: [new docx.MathRun(lower)], superScript: [new docx.MathRun(upper)] }),
  frac: (top, bottom) => new docx.MathFraction({ numerator: top, denominator: bottom }),
  sum: (body, lower, upper) => new docx.MathSum({ children: body, subScript: lower, superScript: upper }),
  sqrt: (body) => new docx.MathRadical({ children: body }),
  paren: (body) => new docx.MathRoundBrackets({ children: body }),
};

// ---------------------------------------------------------------- result data
// Tables in the results chapters are filled from the result files, so the
// numbers in the thesis are never typed by hand.

function readCsv(relativePath) {
  const text = fs.readFileSync(path.join(ROOT, relativePath), "utf8").replace(/^﻿/, "");
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i += 1; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i += 1;
      row.push(field); field = "";
      if (row.length > 1 || row[0] !== "") rows.push(row);
      row = [];
    } else field += c;
  }
  if (field !== "" || row.length > 0) { row.push(field); rows.push(row); }
  const [header, ...body] = rows;
  return body.map((values) => Object.fromEntries(header.map((name, i) => [name, values[i]])));
}

const format = {
  // fixed number of decimals; a true minus sign for negative values
  fixed: (value, decimals = 2) => {
    if (value === "" || value === undefined) return "";
    const text = Number(value).toFixed(decimals);
    return Number(text) === 0 ? (0).toFixed(decimals) : text.replace("-", "−");   // never print "-0.00"
  },
  // fixed decimals with an explicit plus sign for positive values
  signed: (value, decimals = 2) => {
    const text = Number(value).toFixed(decimals);
    if (Number(text) === 0) return (0).toFixed(decimals);
    return (Number(text) > 0 ? "+" : "") + text.replace("-", "−");
  },
  // whole number with thousands separators
  count: (value) => Math.round(Number(value)).toLocaleString("en-US"),
};

module.exports = { buildChapter, math, readCsv, format, ROOT };
