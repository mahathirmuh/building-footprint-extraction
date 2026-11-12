/* its_kit.js - ITS Teknik Informatika lecture-deck kit for pptxgenjs (16:9, 13.333 x 7.5 in).
 *   const { ITSDeck } = require("./its_kit");
 *   const d = new ITSDeck({ assets: "its_assets" });   // folder from its_assets.py, optional
 *   d.cover({...}); const s = d.slide("Judul", "Subjudul"); d.cards(s, [...]); d.closing();
 *   await d.save("deck.pptx");
 * Text fields accept light markup: *italic* (English terms), **bold**, "\n" new paragraph,
 * "- " at line start = bullet. Geometry and colours are measured from the official template.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const zlib = require("zlib");

function resolveModule(name) {                          // pptxgenjs is preinstalled globally, not beside this file
  const roots = [null, ...(process.env.NODE_PATH || "").split(path.delimiter).filter(Boolean), "/opt/npm-tools/node_modules"];
  try { roots.push(require("child_process").execSync("npm root -g", { stdio: ["ignore", "pipe", "ignore"] }).toString().trim()); } catch (e) { /* no npm */ }
  for (const r of roots) { try { return require.resolve(name, r ? { paths: [r] } : undefined); } catch (e) { /* next */ } }
  throw new Error(`Cannot find ${name}. Run node with NODE_PATH set to the node_modules folder that holds pptxgenjs.`);
}
const PPTXGEN_PATH = resolveModule("pptxgenjs");
const pptxgen = require(PPTXGEN_PATH);

const W = 13.333, H = 7.5;
const X0 = 0.723, X1 = 12.617, CW = X1 - X0;          // content column
const TOP = 2.24, TOP_NO_SUB = 1.95, BOTTOM = 6.62, CALLOUT_Y = 6.2, CALLOUT_H = 0.36;
const C = {
  indigo: "302C74", navy: "172F57", ink: "102A43", body: "4B5563", text: "1F2937", muted: "718096",
  line: "D9E2EC", tint: "EAF2FF", tintTeal: "E8F7F7", blue: "0F7BC1", link: "0F9ED5",
  yellow: "FCB914", alert: "C00000", deepTeal: "0E7C86", white: "FFFFFF", paper: "F4F7FB",
};
const ACCENTS = ["16A3A5", "2D6CDF", "F28C45", "7C3AED", "2E8B57", "D64550", "0E7C86"];
const FONT = "Raleway", FONT_BLACK = "Raleway Black";
const DEFAULT_BOX = {                                  // fractions of the slide, from the template
  header_left: [0.053, 0.0551, 0.173, 0.0729], header_right: [0.7485, 0.056, 0.2015, 0.0738],
  footer_on_blue: [0.0565, 0.9529, 0.401, 0.0293], footer_on_white: [0.0565, 0.9529, 0.401, 0.0293],
  hero: [0.045, 0.2587, 0.183, 0.3307], closing_bg: [0, 0, 1, 1],
  closing_header_left: [0.049, 0.0418, 0.1605, 0.0773], closing_header_right: [0.605, 0.056, 0.2015, 0.0738],
  ornament_tr: [0.876, 0, 0.124, 0.216], ornament_bl: [0, 0.7102, 0.1215, 0.2204],
};
const ALIGN = { header_left: "l", header_right: "r", closing_header_left: "l", closing_header_right: "r",
  footer_on_blue: "l", footer_on_white: "l" };

// ---------- small utilities ----------
function imageSize(file) {
  const b = fs.readFileSync(file);
  if (b.readUInt32BE(0) === 0x89504e47) return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
  let i = 2;                                           // JPEG: walk markers to SOFn
  while (i < b.length) {
    const m = b[i + 1], len = b.readUInt16BE(i + 2);
    if (m >= 0xc0 && m <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(m)) return { w: b.readUInt16BE(i + 7), h: b.readUInt16BE(i + 5) };
    i += 2 + len;
  }
  throw new Error("Unsupported image (use PNG or JPG): " + file);
}

function contain(file, x, y, w, h, align = "c") {    // fit an image inside a box, keep aspect
  const { w: iw, h: ih } = imageSize(file);
  const s = Math.min(w / iw, h / ih), dw = iw * s, dh = ih * s;
  const dx = align === "l" ? 0 : align === "r" ? w - dw : (w - dw) / 2;
  return { path: file, x: x + dx, y: y + (h - dh) / 2, w: dw, h: dh };
}

function crc32(buf) {
  let crc = ~0;
  for (const byte of buf) { crc ^= byte; for (let k = 0; k < 8; k++) crc = crc & 1 ? (crc >>> 1) ^ 0xedb88320 : crc >>> 1; }
  return ~crc >>> 0;
}

function fadePng(hex, solidUntil = 0.42, width = 256) {  // horizontal fade, opaque -> transparent
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16));
  const raw = Buffer.alloc(1 + width * 4);
  for (let x = 0; x < width; x++) {
    const t = x / (width - 1), a = t <= solidUntil ? 1 : Math.max(0, 1 - (t - solidUntil) / (1 - solidUntil));
    raw.set([r, g, b, Math.round(255 * a * a * (3 - 2 * a))], 1 + x * 4);
  }
  const chunk = (type, data) => {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type, "latin1"), data]);
    const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(td));
    return Buffer.concat([len, td, crc]);
  };
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(width, 0); ihdr.writeUInt32BE(1, 4); ihdr.set([8, 6, 0, 0, 0], 8);
  const png = Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr),
    chunk("IDAT", zlib.deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]);
  return "image/png;base64," + png.toString("base64");
}

/** Light markup -> pptxgenjs runs. *italic*, **bold**, \n = paragraph, "- " = bullet line. */
function rich(src, base = {}) {
  const out = [];
  const lines = String(src == null ? "" : src).split("\n");
  lines.forEach((raw, li) => {
    const bullet = /^\s*[-•]\s+/.test(raw);
    const line = bullet ? raw.replace(/^\s*[-•]\s+/, "") : raw;
    const start = out.length;
    const re = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*)/g;
    let last = 0, m;
    const push = (text, extra) => { if (text) out.push({ text, options: { ...base, ...extra } }); };
    while ((m = re.exec(line))) {
      push(line.slice(last, m.index), {});
      push(m[0].startsWith("**") ? m[0].slice(2, -2) : m[0].slice(1, -1), m[0].startsWith("**") ? { bold: true } : { italic: true });
      last = m.index + m[0].length;
    }
    push(line.slice(last), {});
    if (out.length === start) out.push({ text: " ", options: { ...base } });
    if (bullet) out[start].options.bullet = { indent: 13 };
    if (li < lines.length - 1) out[out.length - 1].options.breakLine = true;
  });
  return out;
}

function loadAssets(dir) {
  if (!dir || !fs.existsSync(dir)) return {};
  const mf = path.join(dir, "manifest.json");
  const m = fs.existsSync(mf) ? JSON.parse(fs.readFileSync(mf, "utf8")) : {};
  for (const slot of Object.keys(DEFAULT_BOX)) {        // loose files named after a slot also work
    if (m[slot]) continue;
    const f = [".png", ".jpg", ".jpeg"].map((e) => slot + e).find((n) => fs.existsSync(path.join(dir, n)));
    if (f) m[slot] = { file: f, box: DEFAULT_BOX[slot], fit: true };
  }
  const out = {};
  for (const [slot, v] of Object.entries(m)) {
    const file = path.join(dir, v.file);
    if (fs.existsSync(file)) out[slot] = { ...v, path: file, box: v.box || DEFAULT_BOX[slot] };
  }
  return out;
}

function loadJSZip() {                                  // jszip ships with pptxgenjs
  return require(require.resolve("jszip", { paths: [path.dirname(PPTXGEN_PATH)] }));
}

const THEME_COLORS = { dk1: C.ink, lt1: C.white, dk2: C.navy, lt2: C.tint, accent1: ACCENTS[0], accent2: ACCENTS[1],
  accent3: ACCENTS[2], accent4: ACCENTS[3], accent5: ACCENTS[4], accent6: ACCENTS[5], hlink: C.blue, folHlink: C.indigo };

/** Post-process pptxgenjs output: one <a:pPr> per paragraph, ITS theme palette, no folder entries. */
async function finalize(buf) {
  const JSZip = loadJSZip();
  const zin = await JSZip.loadAsync(buf), zout = new JSZip();
  const scheme = '<a:clrScheme name="ITS">' + Object.entries(THEME_COLORS)
    .map(([k, v]) => `<a:${k}><a:srgbClr val="${v}"/></a:${k}>`).join("") + "</a:clrScheme>";
  for (const [name, entry] of Object.entries(zin.files)) {
    if (entry.dir) continue;
    const isXml = /\.(xml|rels)$/.test(name);
    let data = await entry.async(isXml ? "string" : "nodebuffer");
    if (/^ppt\/(slides|slideLayouts|slideMasters)\/[^/]+\.xml$/.test(name)) {
      data = data.replace(/<a:p>([\s\S]*?)<\/a:p>/g, (all, inner) => {
        const i = inner.search(/<a:(r|br|fld)\b/);
        if (i < 0) return all;
        return "<a:p>" + inner.slice(0, i) + inner.slice(i).replace(/<a:pPr\b[^>]*\/>|<a:pPr\b[^>]*>[\s\S]*?<\/a:pPr>/g, "") + "</a:p>";
      });
    }
    if (/^ppt\/theme\/theme\d+\.xml$/.test(name)) {
      data = data.replace(/<a:clrScheme name="[^"]*">[\s\S]*?<\/a:clrScheme>/, scheme).replace(/(<a:theme\b[^>]*name=")[^"]*"/, '$1ITS"');
    }
    zout.file(name, data, { createFolders: false });
  }
  return zout.generateAsync({ type: "nodebuffer", compression: "DEFLATE" });
}

// ---------- the deck ----------
class ITSDeck {
  /**
   * @param {object} o
   *   assets   folder written by its_assets.py (or loose PNGs named after the slots); optional
   *   org      text fallback for the header when there are no logo files
   *   footer   text fallback for the footer strip
   *   lang     proofing language for all text (default "id-ID")
   *   title, author, subject   document properties
   */
  constructor(o = {}) {
    this.a = loadAssets(o.assets);
    this.lang = o.lang || "id-ID";
    this.org = o.org || ["Departemen Teknik Informatika", "Institut Teknologi Sepuluh Nopember"];
    this.footerText = o.footer || "Institut Teknologi Sepuluh Nopember   |   its.ac.id";
    const p = (this.pres = new pptxgen());
    p.layout = "LAYOUT_WIDE";
    p.theme = { headFontFace: FONT, bodyFontFace: FONT };
    if (o.title) p.title = o.title;
    if (o.author) p.author = o.author;
    if (o.subject) p.subject = o.subject;
    this._layouts();
  }

  // ----- primitives (usable for custom slides) -----
  t(o = {}) { return { fontFace: FONT, color: C.body, lang: this.lang, ...o }; }

  text(s, src, box, o = {}) {
    const { size = 15.5, color = C.body, bold, italic, font = FONT, align = "left", valign = "top", spacing } = o;
    s.addText(rich(src, this.t({ fontSize: size, color, bold, italic, fontFace: font })), {
      ...box, margin: 0, align, valign, isTextBox: true, fontFace: font, fontSize: size, color,
      lineSpacing: spacing || Math.round(size * 1.22 * 10) / 10, paraSpaceAfter: o.paraAfter || 0,
      objectName: o.name,
    });
  }

  image(s, file, box, align = "c", alt = "") {
    s.addImage({ ...contain(file, box.x, box.y, box.w, box.h, align), altText: alt });
  }

  placeholder(s, box, label = "Gambar / diagram") {
    s.addText(label, { ...box, fill: { color: C.paper }, line: { color: C.line, width: 0.75, dashType: "dash" },
      shape: this.pres.ShapeType.roundRect, rectRadius: 0.06, fontFace: FONT, fontSize: 12, color: C.muted,
      align: "center", valign: "middle", italic: true, isTextBox: true });
  }

  /** White card with the template's coloured left accent bar. */
  card(s, x, y, w, h, color, head, body, o = {}) {
    const P = this.pres.ShapeType;
    s.addShape(P.roundRect, { x, y, w, h, fill: { color: C.white }, line: { color: C.line, width: 0.75 },
      rectRadius: 0.08, objectName: "Card" });
    s.addShape(P.rect, { x, y, w: 0.073, h, fill: { color }, line: { color, width: 0 }, objectName: "Accent" });
    const hs = o.headSize || 16, bs = o.bodySize || 15.5, padX = o.padX || 0.22;
    const padY = o.padY != null ? o.padY : Math.max(0.12, 0.415 - (0.966 * hs) / 72);
    const runs = [];
    if (head) runs.push(...rich(head, this.t({ fontSize: hs, bold: true, color: C.ink, lineSpacing: hs * 1.2,
      paraSpaceAfter: body ? (o.gap != null ? o.gap : Math.round(bs * 0.55)) : 0 })));
    if (head && body) runs[runs.length - 1].options.breakLine = true;
    if (body) runs.push(...rich(body, this.t({ fontSize: bs, color: o.alert ? C.alert : C.body, lineSpacing: Math.round(bs * 1.22 * 10) / 10 })));
    if (runs.length) s.addText(runs, { x: x + padX, y: y + padY, w: w - padX - 0.18, h: h - padY - 0.12, margin: 0,
      valign: o.valign || "top", isTextBox: true, fontFace: FONT, objectName: "Card text" });
  }

  callout(s, msg, o = {}) {
    const size = o.size || (String(msg).length > 100 ? 12 : 13.5);
    s.addText(rich(msg, this.t({ fontSize: size, bold: true, color: C.ink })), {
      x: X0, y: o.y || CALLOUT_Y, w: CW, h: CALLOUT_H, shape: this.pres.ShapeType.roundRect, rectRadius: 0.04,
      fill: { color: o.tone === "teal" ? C.tintTeal : C.tint }, line: { color: o.tone === "teal" ? C.tintTeal : C.tint, width: 0 },
      margin: [10, 10, 0, 0], align: "center", valign: "middle", isTextBox: true, objectName: "Key message" });
  }

  // ----- layouts (slide masters) -----
  _img(slot, opts = {}) {
    const a = this.a[slot];
    if (!a) return [];
    const [bx, by, bw, bh] = a.box;
    const box = { x: bx * W, y: by * H, w: bw * W, h: bh * H };
    const img = a.fit ? contain(a.path, box.x, box.y, box.w, box.h, ALIGN[slot] || "c") : { path: a.path, ...box };
    return [{ image: { ...img, altText: slot.replace(/_/g, " "), ...opts } }];
  }

  _header(prefix = "") {
    const l = this._img(prefix + "header_left"), r = this._img(prefix + "header_right");
    if (l.length || r.length || prefix) return [...l, ...r];
    return [{ text: { text: this.org[0], options: { x: 7.4, y: 0.42, w: 5.22, h: 0.28, fontFace: FONT, fontSize: 12,
      bold: true, color: C.blue, align: "right", margin: 0 } } },
    { text: { text: this.org[1] || "", options: { x: 7.4, y: 0.7, w: 5.22, h: 0.24, fontFace: FONT, fontSize: 10,
      color: C.muted, align: "right", margin: 0 } } }];
  }

  _footer(onBlue) {
    const img = this._img(onBlue ? "footer_on_blue" : "footer_on_white");
    if (img.length) return img;
    return [{ text: { text: this.footerText, options: { x: 0.754, y: 7.1, w: 5.4, h: 0.3, fontFace: FONT, fontSize: 9.5,
      color: onBlue ? C.white : C.blue, margin: 0, valign: "middle" } } }];
  }

  _band(color) {
    return [{ text: { text: "", options: { shape: this.pres.ShapeType.round1Rect, x: 0, y: 6.98, w: 6.426, h: 0.52,
      rectRadius: 0.229, fill: { color }, line: { color, width: 0 } } } }];
  }

  _layouts() {
    const p = this.pres;
    p.defineSlideMaster({
      title: "ITS_TITLE", background: { color: C.white },
      objects: [...this._header(), ...this._footer(false)],
      slideNumber: { x: 12.55, y: 7.08, w: 0.5, h: 0.22, fontFace: FONT, fontSize: 8, color: C.navy, align: "right", margin: 0 },
    });
    p.defineSlideMaster({
      title: "ITS_CONTENT", background: { color: C.white },
      objects: [...this._header(), ...this._band(C.blue), ...this._footer(true),
        { placeholder: { options: { name: "title", type: "title", x: X0 - 0.067, y: 1.171, w: CW, h: 0.52, fontFace: FONT,
          fontSize: 27, bold: true, color: C.navy, align: "left", valign: "middle", margin: 0 }, text: "" } }],
      slideNumber: { x: 12.55, y: 7.08, w: 0.5, h: 0.22, fontFace: FONT, fontSize: 8.5, color: C.muted, align: "right", margin: 0 },
    });
    const bg = this.a.closing_bg;
    p.defineSlideMaster({
      title: "ITS_CLOSING", background: { color: C.blue },
      objects: [
        ...(bg ? [{ image: { path: bg.path, x: 0, y: 0, w: W, h: H, transparency: Math.round(100 * (1 - (bg.opacity || 0.2))) } }] : []),
        ...(bg ? [{ image: { data: fadePng(C.blue), x: 0, y: 0, w: W * 0.6, h: H } }] : []),
        ...this._img("ornament_tr"), ...this._img("ornament_bl"), ...this._header("closing_"),
        ...this._band(C.white), ...this._footer(false),
        { text: { text: "Institut Teknologi Sepuluh Nopember", options: { x: 7.0, y: 7.12, w: 5.59, h: 0.3, fontFace: FONT,
          fontSize: 15, color: C.white, align: "right", margin: 0, valign: "middle" } } },
        { rect: { x: 12.693, y: 6.98, w: 0.64, h: 0.52, fill: { color: C.yellow }, line: { color: C.yellow, width: 0 } } },
      ],
      slideNumber: { x: 12.693, y: 7.05, w: 0.64, h: 0.3, fontFace: FONT, fontSize: 8, color: C.navy, align: "center", margin: 0 },
    });
  }

  // ----- whole slides -----
  _add(master, section) {
    if (section) {
      let t = String(section).split("\n")[0].slice(0, 60), k = 2;
      while ((this._sections || []).includes(t)) t = `${String(section).split("\n")[0].slice(0, 56)} (${k++})`;
      (this._sections = this._sections || []).push(t);
      this.pres.addSection({ title: t });
      this._sect = t;
    }
    return this.pres.addSlide(this._sect ? { masterName: master, sectionTitle: this._sect } : { masterName: master });
  }

  _pill(s, text, x, y, size, color, padX) {
    const est = String(text).length * size * 0.0072 + 2 * padX;
    const w = Math.min(9.3, Math.max(5.906, est));
    s.addText(rich(text, this.t({ fontSize: size, bold: true, italic: true, color })), { x, y, w, h: 0.706,
      shape: this.pres.ShapeType.roundRect, rectRadius: 0.265, fill: { color: C.yellow }, line: { color: C.yellow, width: 0 },
      margin: [Math.max(0.02, padX - 0.0776) * 72, 0.1 * 72, 0, 0], valign: "middle", isTextBox: true, objectName: "Pill" });
  }

  _hero(s, file, box) {
    if (file && fs.existsSync(file)) this.image(s, file, box, "c", "Logo / ikon");
  }

  /** Program cover: indigo Raleway Black title, yellow pill, presenter bottom-right. */
  cover({ title, pill, name, email, hero } = {}) {
    const s = this._add("ITS_TITLE", "Pembuka");
    const heroFile = hero || (this.a.hero && this.a.hero.path);
    this._hero(s, heroFile, this.a.hero && !hero ? { x: this.a.hero.box[0] * W, y: this.a.hero.box[1] * H,
      w: this.a.hero.box[2] * W, h: this.a.hero.box[3] * H } : { x: 0.6, y: 1.94, w: 2.44, h: 2.48 });
    const x = heroFile ? 3.579 : X0;
    const lines = Array.isArray(title) ? title.join("\n") : title || "";
    this.text(s, lines, { x, y: 2.647, w: W - x - 0.5, h: 1.4 }, { size: 44, font: FONT_BLACK, color: C.indigo,
      spacing: 44.2, name: "Cover title" });
    if (pill) this._pill(s, pill, x + 0.134, 4.1, 17.3, C.indigo, 0.247);
    const who = [];
    if (name) who.push({ text: name, options: this.t({ fontSize: 13.5, bold: true, color: C.indigo, lineSpacing: 21.4, breakLine: !!email }) });
    if (email) who.push({ text: email, options: this.t({ fontSize: 12.5, italic: true, color: C.indigo, lineSpacing: 21.4 }) });
    if (who.length) s.addText(who, { x: 6.6, y: 6.238, w: 5.986, h: 0.62, margin: 0, align: "right", valign: "top",
      isTextBox: true, objectName: "Presenter" });
    return s;
  }

  /** Course / topic title (the template's 2nd slide). Doubles as a section divider. */
  section({ title, pill, label, hero } = {}) {
    const s = this._add("ITS_TITLE", label || (Array.isArray(title) ? title.join(" ") : title));
    const heroFile = hero || (this.a.hero && this.a.hero.path);
    this._hero(s, heroFile, { x: 0.806, y: 2.073, w: 2.1, h: 2.1 });
    const x = heroFile ? 3.579 : X0;
    const lines = Array.isArray(title) ? title.join("\n") : title || "";
    this.text(s, lines, { x, y: 2.779, w: W - x - 0.5, h: 1.25 }, { size: 36, font: FONT_BLACK, color: C.navy,
      spacing: 39, name: "Section title" });
    if (pill) this._pill(s, pill, x + 0.134, 4.1, 21, C.navy, 0.17);
    if (label) this.text(s, String(label).toUpperCase(), { x: 5.5, y: 6.029, w: 7.026, h: 0.4 }, { size: 18.7, bold: true,
      color: C.navy, align: "right", spacing: 22.4, name: "Topic" });
    return s;
  }

  /** Content slide: title (placeholder) + optional grey subtitle. Content area y 2.24 (1.95 without subtitle) to 6.62 in. */
  slide(title, subtitle, o = {}) {
    const s = this._add("ITS_CONTENT");
    const size = String(title).length > 62 ? 23 : 27;
    s.addText(rich(title, this.t({ fontSize: size, bold: true, color: C.navy })), { placeholder: "title" });
    if (subtitle) this.text(s, subtitle, { x: X0 - 0.067, y: 1.70, w: CW + 0.07, h: 0.36 }, { size: 16, name: "Subtitle" });
    if (o.notes) s.addNotes(o.notes);
    s._itsTop = subtitle ? TOP : TOP_NO_SUB;
    return s;
  }

  /** Closing slide ("Terima Kasih!"). */
  closing(text = "Terima Kasih!", o = {}) {
    const s = this._add("ITS_CLOSING", this._sect ? "Penutup" : null);
    s.addText("“", { x: 1.29, y: 1.75, w: 1.6, h: 1.6, fontFace: FONT_BLACK, fontSize: 110, color: C.white, margin: 0, valign: "top", isTextBox: true });
    s.addText("”", { x: 10.2, y: 3.32, w: 1.6, h: 1.6, fontFace: FONT_BLACK, fontSize: 110, color: C.white, margin: 0, valign: "top", align: "right", isTextBox: true });
    this.text(s, text, { x: 2.0, y: 3.22, w: W - 4.0, h: 0.9 }, { size: 40, bold: true, color: C.white, align: "center",
      valign: "middle", name: "Closing" });
    if (o.sub) this.text(s, o.sub, { x: 2.0, y: 4.12, w: W - 4.0, h: 0.5 }, { size: 16, color: C.white, align: "center" });
    return s;
  }

  // ----- content builders (all take the slide returned by slide()) -----
  /** fixed=true for builders with a set row pitch (they shrink the pitch to fit above a callout). */
  _area(o, s, fixed = false) {
    const bottom = o.bottom || (o.callout ? (fixed ? 6.27 : 5.85) : BOTTOM);
    return { top: o.top || (s && s._itsTop) || TOP, bottom };
  }

  /** Key-message bar sits 0.22 in under the content, between y 6.2 and 6.49 (as in the template). */
  _done(s, o, contentBottom) {
    if (o.callout) this.callout(s, o.callout, { tone: o.tone, y: o.calloutY || Math.min(6.49, Math.max(6.2, contentBottom + 0.22)) });
    return s;
  }

  /**
   * Card grid. items: [{head, body, color?, alert?}]. rows: e.g. [3], [3,2], [4,3]; a shorter row is centred.
   * o: { rows, callout, tone, top, bottom, headSize, bodySize, gap }
   */
  cards(s, items, o = {}) {
    const n = items.length;
    const rows = o.rows || (n <= 4 ? [n] : n === 5 ? [3, 2] : n === 6 ? [3, 3] : [4, n - 4]);
    const { top, bottom } = this._area(o, s);
    const cols = Math.max(...rows), hg = o.gap || (rows.length > 1 ? 0.3 : 0.18), vg = 0.3;
    const cw = (CW - hg * (cols - 1)) / cols, ch = (bottom - top - vg * (rows.length - 1)) / rows.length;
    const dense = n >= 6 || rows.length > 1 && ch < 1.9;
    let k = 0;
    rows.forEach((count, r) => {
      const x0 = X0 + (CW - (count * cw + (count - 1) * hg)) / 2;
      for (let c = 0; c < count && k < n; c++, k++) {
        const it = items[k];
        this.card(s, x0 + c * (cw + hg), top + r * (ch + vg), cw, ch, it.color || ACCENTS[k % ACCENTS.length], it.head, it.body,
          { alert: it.alert, headSize: o.headSize || (rows.length > 1 ? 18.5 : 16), bodySize: o.bodySize || (dense ? 14.5 : 15.5) });
      }
    });
    return this._done(s, o, bottom);
  }

  /** Visual panel left + stacked cards right. o: { image, label, items:[{head, body, alert?}], split, callout } */
  mediaStack(s, o) {
    const { top, bottom } = this._area(o, s);
    const items = o.items || [], n = items.length;
    const split = o.split || 0.55, gap = 0.28, pw = n ? CW * split - gap / 2 : CW;
    const panel = { x: X0, y: top, w: pw, h: bottom - top };
    s.addShape(this.pres.ShapeType.roundRect, { ...panel, fill: { color: C.white }, line: { color: C.line, width: 0.75 },
      rectRadius: 0.06, objectName: "Visual panel" });
    const inner = { x: panel.x + 0.15, y: panel.y + 0.15, w: panel.w - 0.3, h: panel.h - 0.3 };
    if (o.image) this.image(s, o.image, inner, "c", o.alt || o.label || "");
    else this.placeholder(s, inner, o.label);
    const vg = 0.2;
    const rx = X0 + pw + gap, rw = X1 - rx, rh = (bottom - top - vg * (n - 1)) / n;
    items.forEach((it, i) => this.card(s, rx, top + i * (rh + vg), rw, rh, it.color || ACCENTS[i % 3 === 1 ? 2 : i % 3 === 2 ? 1 : 0],
      it.head, it.body, { alert: it.alert, headSize: n > 3 ? 15 : 16.5, bodySize: n > 3 ? 14 : 15 }));
    return this._done(s, o, bottom);
  }

  /** Numbered rows: badge + bold label + description. o: { highlight (1-based), shape: "circle"|"square", labelW, callout } */
  numbered(s, items, o = {}) {
    const { top, bottom } = this._area(o, s, true);
    const n = items.length, sq = o.shape === "square";
    const d = sq ? 0.62 : 0.48, pitch = Math.min(0.933, (bottom - top - d) / Math.max(1, n - 1));
    const bx = sq ? 0.696 : 0.823, lx = bx + d + (sq ? 0.108 : 0.25), lw = o.labelW || (sq ? 4.2 : 2.3);
    const dx = lx + lw + 0.12;
    items.forEach((it, i) => {
      const cy = top + i * pitch + d / 2, color = o.highlight === i + 1 ? ACCENTS[2] : ACCENTS[0];
      s.addText(String(i + 1), { x: bx, y: cy - d / 2, w: d, h: d, shape: sq ? this.pres.ShapeType.roundRect : this.pres.ShapeType.ellipse,
        rectRadius: sq ? 0.08 : undefined, fill: { color }, line: { color, width: 0 }, fontFace: FONT, fontSize: sq ? 14 : 12,
        bold: true, color: C.white, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "Badge" });
      this.text(s, it.head, { x: lx, y: cy - 0.32, w: lw, h: 0.64 }, { size: sq ? 15.9 : 16.8, bold: true, color: C.ink, valign: "middle" });
      if (it.body) this.text(s, it.body, { x: dx, y: cy - 0.36, w: X1 - dx, h: 0.72 }, { size: sq ? 15 : 14.5, valign: "middle" });
    });
    return this._done(s, o, top + (n - 1) * pitch + d);
  }

  /** Pipeline rows: teal step label - connector - description. o: { highlight (1-based), callout } */
  pipeline(s, steps, o = {}) {
    const { top, bottom } = this._area(o, s, true);
    const n = steps.length, pitch = Math.min(0.694, (bottom - top - 0.4) / Math.max(1, n - 1));
    steps.forEach((st, i) => {
      const y = top - 0.06 + i * pitch, color = o.highlight === i + 1 ? ACCENTS[2] : C.deepTeal;
      this.text(s, `${i + 1}. ${st.head}`, { x: 0.85, y, w: 3.05, h: 0.4 }, { size: 15.9, bold: true, color, valign: "middle" });
      s.addShape(this.pres.ShapeType.line, { x: 3.973, y: y + 0.2, w: 0.5, h: 0, line: { color: C.link, width: 1 } });
      if (st.body) this.text(s, st.body, { x: 4.705, y, w: X1 - 4.705, h: 0.4 }, { size: 15.9, valign: "middle" });
    });
    return this._done(s, o, top - 0.06 + (n - 1) * pitch + 0.4);
  }

  /** Era/phase cards in up to two rows with chevrons. items: [{head, body}] (max 8). */
  timeline(s, items, o = {}) {
    const { top, bottom } = this._area(o, s);
    const n = items.length, r1 = Math.min(4, n), r2 = n - r1, gap = 0.24, vg = 0.33;
    const rh = r2 ? (bottom - top - vg) / 2 : Math.min(2.6, bottom - top);
    const row = (count, offset, y) => {
      const cw = count === r1 && r2 ? (CW - gap * (r1 - 1)) / r1 : Math.min(3.18, (CW - gap * (count - 1)) / count);
      const x0 = X0 + (CW - (count * cw + (count - 1) * gap)) / 2;
      for (let i = 0; i < count; i++) {
        const k = offset + i, x = x0 + i * (cw + gap);
        this.card(s, x, y, cw, rh, items[k].color || ACCENTS[k % ACCENTS.length], items[k].head, items[k].body,
          { headSize: 14.5, bodySize: 14, padX: 0.2, gap: 6 });
        if (i < count - 1) s.addShape(this.pres.ShapeType.chevron, { x: x + cw - 0.02, y: y + rh * 0.3, w: 0.28, h: 0.45,
          fill: { color: C.line }, line: { color: C.line, width: 0 } });
      }
      return { x0, cw };
    };
    const a = row(r1, 0, top);
    if (r2) {
      const b = row(r2, r1, top + rh + vg);
      const cx = Math.min(a.x0 + (r1 - 0.5) * a.cw + (r1 - 1) * gap, b.x0 + r2 * b.cw + (r2 - 1) * gap - 0.4);
      s.addShape(this.pres.ShapeType.rect, { x: cx, y: top + rh + 0.03, w: 0.013, h: vg - 0.06, fill: { color: C.line }, line: { color: C.line, width: 0 } });
      s.addShape(this.pres.ShapeType.ellipse, { x: cx - 0.021, y: top + rh + vg - 0.06, w: 0.055, h: 0.055, fill: { color: C.line }, line: { color: C.line, width: 0 } });
    }
    return this._done(s, o, bottom);
  }

  /** Numbered reference list (APA-style strings). o: { start, callout } */
  references(s, refs, o = {}) {
    const { top, bottom } = this._area(o, s, true);
    const pitch = Math.min(0.72, (bottom - 0.345 - top) / Math.max(1, refs.length - 1));
    refs.forEach((r, i) => {
      const y = top + 0.055 + i * pitch;
      this.text(s, `${(o.start || 1) + i}.`, { x: 0.82, y, w: 0.4, h: pitch - 0.05 }, { size: 14, bold: true, color: C.deepTeal });
      this.text(s, r, { x: 1.22, y, w: X1 - 1.22, h: pitch - 0.05 }, { size: 14, color: C.text });
    });
    return this._done(s, { ...o, tone: o.tone || "teal" }, top + 0.055 + (refs.length - 1) * pitch + 0.25);
  }

  /** Simple table in ITS colours. header: [..], rows: [[..]], o: { colW, size, callout } */
  table(s, header, rows, o = {}) {
    const { top } = this._area(o, s);
    const size = o.size || 13;
    const cell = (v, i, head, r) => ({ text: rich(v, this.t({ fontSize: size, bold: head || undefined, color: head ? C.white : C.text })),
      options: { fill: { color: head ? C.navy : r % 2 ? C.tint : C.white }, valign: "middle", align: i === 0 ? "left" : (o.align || "left"),
        border: [{ type: "solid", color: C.line, pt: 0.75 }, { type: "none" }, { type: "solid", color: C.line, pt: 0.75 }, { type: "none" }],
        margin: [4, 8, 4, 8] } });
    const data = [header.map((h, i) => cell(h, i, true, 0)), ...rows.map((r, ri) => r.map((v, i) => cell(v, i, false, ri)))];
    const rowH = o.rowH || 0.42;
    s.addTable(data, { x: X0, y: top, w: CW, colW: o.colW, rowH, fontFace: FONT, autoPage: false });
    return this._done(s, o, top + data.length * rowH);
  }

  /** Native chart in the accent palette. type: "bar"|"line"|"pie"|"doughnut"; data: pptxgenjs chart data. */
  chart(s, type, data, o = {}) {
    const { top, bottom } = this._area(o, s);
    const box = o.box || { x: X0, y: top, w: CW, h: bottom - top };
    const t = { bar: this.pres.charts.BAR, line: this.pres.charts.LINE, pie: this.pres.charts.PIE, doughnut: this.pres.charts.DOUGHNUT }[type];
    const single = data.length === 1 && !/pie|doughnut/.test(type);   // one series = one colour
    s.addChart(t, data, { ...box, chartColors: o.colors || (single ? [ACCENTS[0]] : ACCENTS), barDir: o.barDir || "col",
      dataLabelFormatCode: o.format || "General", valAxisLabelFormatCode: o.axisFormat || "General",
      catAxisLabelFontFace: FONT, valAxisLabelFontFace: FONT, legendFontFace: FONT, dataLabelFontFace: FONT, titleFontFace: FONT,
      catAxisLabelColor: C.body, valAxisLabelColor: C.body, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
      valGridLine: { color: "E5EAF0", size: 0.75 }, catGridLine: { style: "none" }, showLegend: data.length > 1 || /pie|doughnut/.test(type),
      legendPos: "b", legendFontSize: 12, legendColor: C.body, showValue: o.showValue !== false, dataLabelFontSize: 11,
      dataLabelColor: C.ink, showTitle: !!o.title, title: o.title, titleColor: C.ink, titleFontSize: 14, ...(o.extra || {}) });
    return this._done(s, o, box.y + box.h);
  }

  async save(file) {
    const buf = await this.pres.write({ outputType: "nodebuffer" });
    fs.writeFileSync(file, await finalize(buf));
    return file;
  }
}

module.exports = { ITSDeck, C, ACCENTS, FONT, FONT_BLACK, W, H, X0, X1, CW, rich, contain, imageSize };
