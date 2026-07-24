import { beforeEach, describe, expect, it, vi } from "vitest";
import { loadListingFonts, loadListingImages } from "@/lib/listing/assets";

const fakeEncode = async (blob: Blob) => `data:image/jpeg;base64,${(blob as never as { tag: string }).tag}`;

function blobFor(tag: string): Blob {
  const b = new Blob(["x"], { type: "image/jpeg" });
  (b as never as { tag: string }).tag = tag;
  return b;
}

describe("loadListingImages", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("loads every image in index order", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      const n = url.slice(url.lastIndexOf("/") + 1);
      return { ok: true, blob: async () => blobFor(`I${n}`) } as unknown as Response;
    }));

    const images = await loadListingImages("s1", 3, fakeEncode);

    expect(images.map((i) => i.index)).toEqual([0, 1, 2]);
    expect(images[2].dataUri).toContain("I2");
  });

  it("skips an image whose request fails and keeps the real indices", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      const n = url.slice(url.lastIndexOf("/") + 1);
      if (n === "1") return { ok: false, status: 404 } as unknown as Response;
      return { ok: true, blob: async () => blobFor(`I${n}`) } as unknown as Response;
    }));

    const images = await loadListingImages("s1", 3, fakeEncode);

    expect(images.map((i) => i.index)).toEqual([0, 2]);
  });

  it("returns an empty array when every request fails", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 404 }) as unknown as Response));
    expect(await loadListingImages("s1", 2, fakeEncode)).toEqual([]);
  });

  it("survives a rejected request without throwing", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("network down"); }));
    expect(await loadListingImages("s1", 2, fakeEncode)).toEqual([]);
  });

  it("does not fetch anything when the count is zero", async () => {
    const spy = vi.fn();
    vi.stubGlobal("fetch", spy);
    expect(await loadListingImages("s1", 0, fakeEncode)).toEqual([]);
    expect(spy).not.toHaveBeenCalled();
  });
});

describe("loadListingFonts", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("returns null when a font cannot be fetched, rather than failing the export", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 404 }) as unknown as Response));
    expect(await loadListingFonts()).toBeNull();
  });

  it("loads both fonts and keeps display/body matched to their own path", async () => {
    vi.stubGlobal("fetch", vi.fn(async (path: string) => {
      const body = path.includes("fraunces") ? "FRAUNCES_BYTES" : "GEIST_BYTES";
      return { ok: true, blob: async () => new Blob([body], { type: "font/woff2" }) } as unknown as Response;
    }));

    const fonts = await loadListingFonts();

    expect(fonts).not.toBeNull();
    expect(fonts?.display).toContain(btoa("FRAUNCES_BYTES"));
    expect(fonts?.body).toContain(btoa("GEIST_BYTES"));
  });
});
