import { reportImageUrl } from "@/lib/images";
import type { ListingFonts, ListingImage } from "./buildListingHtml";

const MAX_EDGE = 1200;
const JPEG_QUALITY = 0.82;

export type Encoder = (blob: Blob) => Promise<string>;

async function toDataUri(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

/**
 * Downscale an image blob to MAX_EDGE on its longest side and re-encode it as
 * JPEG. Uses <canvas>, so it cannot run under jsdom; loadListingImages accepts
 * a replacement so its own logic stays testable. Throws if a 2D canvas
 * context is unavailable rather than silently falling back to the original
 * blob; loadListingImages' per-image try/catch turns that into a skipped
 * image.
 */
export async function downscaleToDataUri(blob: Blob): Promise<string> {
  const bitmap = await createImageBitmap(blob);
  const scale = Math.min(1, MAX_EDGE / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  const ctx = canvas.getContext("2d");
  if (ctx === null) throw new Error("Failed to acquire a 2D canvas context for image downscaling");
  ctx.fillStyle = "#FFFFFF";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  return canvas.toDataURL("image/jpeg", JPEG_QUALITY);
}

/**
 * Fetch and inline every uploaded photo. A photo that cannot be retrieved or
 * encoded is skipped; the surviving photos keep their original upload index
 * so the hero selection in buildListingHtml stays correct.
 */
export async function loadListingImages(
  sessionId: string,
  count: number,
  encode: Encoder = downscaleToDataUri,
): Promise<ListingImage[]> {
  const results = await Promise.all(
    Array.from({ length: count }, async (_unused, index): Promise<ListingImage | null> => {
      try {
        const res = await fetch(reportImageUrl(sessionId, index), { cache: "no-store" });
        if (!res.ok) return null;
        return { index, dataUri: await encode(await res.blob()) };
      } catch {
        return null;
      }
    }),
  );
  return results.filter((r): r is ListingImage => r !== null);
}

/** Inline the two artifact fonts. Returns null if either is unavailable. */
export async function loadListingFonts(): Promise<ListingFonts | null> {
  try {
    const [display, body] = await Promise.all(
      ["/fonts/fraunces.woff2", "/fonts/geist.woff2"].map(async (path) => {
        const res = await fetch(path, { cache: "force-cache" });
        if (!res.ok) throw new Error(path);
        return toDataUri(await res.blob());
      }),
    );
    return { display, body };
  } catch {
    return null;
  }
}
