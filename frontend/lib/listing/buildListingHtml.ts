import type { AdoptionReport, PetProfileRequest } from "@/lib/types";
import type { SpeedClass } from "@/lib/spectrum";
import type { ListingLabels } from "./labels";
import { escapeHtml } from "./escape";

export interface ListingImage { index: number; dataUri: string; }
export interface ListingFonts { display: string; body: string; }

export interface ListingInput {
  report: AdoptionReport;
  profile: PetProfileRequest;
  labels: ListingLabels;
  images: ListingImage[];
  fonts: ListingFonts | null;
  classes: SpeedClass[];
}

const INK = "#17241D";
const MUTED = "#5B6660";
const PAPER = "#E8E6DD";
const SURFACE = "#FBFAF6";
const TEAL = "#0E7C7B";

function fontFaces(fonts: ListingFonts | null): string {
  if (!fonts) return "";
  // These src values are base64 data URIs produced by readAsDataURL, so they
  // never contain quotes or angle brackets. That controlled input, not the
  // escapeHtml call, is what keeps this url() safe: HTML entities are not
  // decoded inside a <style> raw-text element. The escape is defense in depth.
  return `
@font-face { font-family: "ListingDisplay"; src: url("${escapeHtml(fonts.display)}") format("woff2"); font-display: swap; }
@font-face { font-family: "ListingBody"; src: url("${escapeHtml(fonts.body)}") format("woff2"); font-display: swap; }`;
}

function styles(fonts: ListingFonts | null): string {
  return `
${fontFaces(fonts)}
* { box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin: 0; background: ${PAPER}; color: ${INK};
  font-family: "ListingBody", ui-sans-serif, system-ui, sans-serif; }
.page { background: ${SURFACE}; max-width: 820px; margin: 24px auto; padding: 40px; }
.page + .page { margin-top: 32px; }
h1, h2 { font-family: "ListingDisplay", Georgia, "Times New Roman", serif; font-weight: 600; margin: 0; }
h1 { font-size: 42px; line-height: 1.1; }
h2 { font-size: 13px; letter-spacing: 0.16em; text-transform: uppercase; color: ${MUTED}; }
.spread { display: flex; gap: 32px; align-items: flex-start; }
.col-left { width: 38%; flex-shrink: 0; }
.col-right { width: 62%; }
.hero { width: 100%; aspect-ratio: 4 / 5; object-fit: cover; display: block; background: ${PAPER}; }
.hero-blank { width: 100%; aspect-ratio: 4 / 5; background: ${PAPER};
  display: flex; align-items: center; justify-content: center;
  font-family: "ListingDisplay", Georgia, serif; font-size: 20px; color: ${MUTED}; text-align: center; padding: 16px; }
.rule { height: 2px; background: ${TEAL}; width: 64px; margin: 16px 0 20px; }
.spec { font-size: 14px; color: ${MUTED}; line-height: 1.7; margin-top: 16px; }
.rows { margin-top: 20px; border-top: 1px solid ${PAPER}; font-size: 13px; }
.row { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid ${PAPER}; }
.row span:last-child { color: ${MUTED}; }
.body-copy { font-size: 15px; line-height: 1.75; white-space: pre-wrap; }
.note { font-size: 12px; color: ${MUTED}; font-style: italic; margin-bottom: 10px; }
.thumbs { display: flex; gap: 10px; margin-top: 26px; }
.thumbs img { width: 92px; height: 92px; object-fit: cover; }
.brief-row { display: flex; gap: 28px; margin-top: 22px; }
.brief-col { flex: 1; }
.brief-label { font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: ${MUTED}; margin-bottom: 8px; }
.verdict { font-family: "ListingDisplay", Georgia, serif; font-size: 24px; margin-top: 14px; }
.conf { font-size: 13px; color: ${MUTED}; }
.spectrum { display: flex; gap: 3px; margin-top: 14px; }
.seg { flex: 1; height: 10px; opacity: 0.28; }
.seg.marker { opacity: 1; height: 16px; margin-top: -3px; }
.seg-labels { display: flex; gap: 3px; margin-top: 6px; font-size: 9px; color: ${MUTED}; }
.seg-labels span { flex: 1; text-align: center; }
.actions { margin-top: 26px; font-size: 14px; }
.action { display: flex; gap: 12px; padding: 9px 0; border-bottom: 1px solid ${PAPER}; }
.action b { font-weight: 600; }
.action i { color: ${TEAL}; font-style: normal; white-space: nowrap; }
.footer { margin-top: 28px; font-size: 11px; color: ${MUTED}; letter-spacing: 0.08em; }
@page { size: A4; margin: 14mm; }
@media print {
  body { background: ${SURFACE}; }
  .page { margin: 0; max-width: none; padding: 0; }
  .page + .page { break-before: page; margin-top: 0; }
}`;
}

function specLine(l: ListingLabels): string {
  return [l.species, l.breed, l.age, l.gender, l.size, l.fur && `${l.fur} fur`, l.colors, l.state]
    .filter((part): part is string => Boolean(part))
    .map(escapeHtml)
    .join(" &middot; ");
}

function renderHero(input: ListingInput): string {
  const { images, report, labels } = input;
  if (images.length === 0) {
    return `<div class="hero-blank">${escapeHtml(labels.title)}</div>`;
  }
  const best = report.visual?.best_photo_index ?? 0;
  const hero = images.find((i) => i.index === best) ?? images[0];
  return `<img class="hero" src="${escapeHtml(hero.dataUri)}" alt="${escapeHtml(labels.title)}">`;
}

function renderThumbs(input: ListingInput): string {
  const { images, report } = input;
  if (images.length < 2) return "";
  const best = report.visual?.best_photo_index ?? 0;
  const heroIndex = images.some((i) => i.index === best) ? best : images[0].index;
  const rest = images.filter((i) => i.index !== heroIndex);
  if (rest.length === 0) return "";
  const tags = rest
    .map((i) => `<img src="${escapeHtml(i.dataUri)}" alt="Photo ${i.index + 1}">`)
    .join("");
  return `<div class="thumbs">${tags}</div>`;
}

/** The description shown in the ad, plus whether it is a labelled fallback. */
function resolveDescription(input: ListingInput): { text: string; isFallback: boolean } | null {
  const optimized = (input.report.optimized_description ?? "").trim();
  if (optimized) return { text: optimized, isFallback: false };
  const original = (input.profile.description ?? "").trim();
  if (original) return { text: original, isFallback: true };
  return null;
}

function renderAd(input: ListingInput): string {
  const l = input.labels;
  const description = resolveDescription(input);
  const rows = l.health
    .map((h) => `<div class="row"><span>${escapeHtml(h.label)}</span><span>${escapeHtml(h.value)}</span></div>`)
    .join("");

  const copy = description
    ? `${description.isFallback
        ? `<p class="note">Original description &mdash; the rewrite was unavailable for this run.</p>`
        : ""}<p class="body-copy">${escapeHtml(description.text)}</p>`
    : "";

  return `<section class="page">
  <div class="spread">
    <div class="col-left">
      ${renderHero(input)}
      <div class="spec">${specLine(l)}</div>
      <div class="rows">${rows}<div class="row"><span>Fee</span><span>${escapeHtml(l.fee)}</span></div></div>
    </div>
    <div class="col-right">
      <h1>${escapeHtml(l.title)}</h1>
      <div class="rule"></div>
      ${copy}
      ${renderThumbs(input)}
    </div>
  </div>
</section>`;
}

function renderSpectrum(input: ListingInput): string {
  const predicted = input.report.prediction.predicted_class;
  const segs = input.classes
    .map(
      (c) =>
        `<div class="seg${c.index === predicted ? " marker" : ""}" style="background:${c.color}"></div>`,
    )
    .join("");
  const names = input.classes
    .map((c) => `<span>${escapeHtml(c.label)}</span>`)
    .join("");
  return `<div class="spectrum">${segs}</div><div class="seg-labels">${names}</div>`;
}

function renderComparison(input: ListingInput): string {
  const optimized = (input.report.optimized_description ?? "").trim();
  const original = (input.profile.description ?? "").trim();

  if (!optimized) {
    return `<p class="note">No rewritten description was produced for this run, so there is nothing to compare.</p>`;
  }
  return `<div class="brief-row">
    <div class="brief-col">
      <div class="brief-label">Before</div>
      <p class="body-copy">${original ? escapeHtml(original) : "No original description was provided."}</p>
    </div>
    <div class="brief-col">
      <div class="brief-label">After</div>
      <p class="body-copy">${escapeHtml(optimized)}</p>
    </div>
  </div>`;
}

function renderActions(input: ListingInput): string {
  const recs = input.report.recommendations?.recommendations ?? [];
  if (recs.length === 0) return "";
  const top = [...recs].sort((a, b) => a.priority - b.priority).slice(0, 3);
  const rows = top
    .map(
      (r) =>
        `<div class="action"><b>${escapeHtml(r.action)}</b><i>${escapeHtml(
          r.measured_impact?.expected_speedup ?? "",
        )}</i></div>`,
    )
    .join("");
  return `<h2 style="margin-top:30px">What would move the needle</h2><div class="actions">${rows}</div>`;
}

function renderBrief(input: ListingInput): string {
  const p = input.report.prediction;
  const meta = input.report.metadata;
  const date = (meta.timestamp ?? "").slice(0, 10);
  return `<section class="page">
  <h2>Why this works</h2>
  <div class="verdict">${escapeHtml(p.prediction_label)}</div>
  <div class="conf">Confidence ${Math.round((p.class_confidence ?? 0) * 100)}%</div>
  ${renderSpectrum(input)}
  ${renderComparison(input)}
  ${renderActions(input)}
  <div class="footer">${escapeHtml(meta.ml_model_version ?? "")}${date ? ` &middot; ${escapeHtml(date)}` : ""}</div>
</section>`;
}

export function buildListingHtml(input: ListingInput): string {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(input.labels.title)}</title>
<style>${styles(input.fonts)}</style>
</head>
<body>
${renderAd(input)}
${renderBrief(input)}
</body>
</html>`;
}
