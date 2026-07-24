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

/** Page 2 is added in the next task. */
function renderBrief(_input: ListingInput): string {
  return "";
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
