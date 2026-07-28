"use client";
import { useEffect, useMemo, useState } from "react";
import type { AdoptionReport, PetProfileRequest } from "@/lib/types";
import { useMeta } from "@/lib/useMeta";
import { buildSpeedClasses } from "@/lib/spectrum";
import { buildListingLabels } from "@/lib/listing/labels";
import { buildListingHtml, type ListingFonts, type ListingImage } from "@/lib/listing/buildListingHtml";
import { loadListingFonts, loadListingImages } from "@/lib/listing/assets";

function fileName(listing: PetProfileRequest, sessionId: string): string {
  const slug =
    (listing.name ?? "")
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "") || "pet";
  return `listing-${slug}-${sessionId.slice(0, 8)}.html`;
}

export function ListingPreview({
  report,
  listing,
}: {
  report: AdoptionReport;
  listing: PetProfileRequest;
}) {
  const { data: meta } = useMeta();
  const [images, setImages] = useState<ListingImage[] | null>(null);
  const [fonts, setFonts] = useState<ListingFonts | null>(null);

  const sessionId = report.metadata.session_id;
  const imageCount = report.metadata.image_count ?? 0;

  useEffect(() => {
    let active = true;
    void Promise.all([loadListingImages(sessionId, imageCount), loadListingFonts()]).then(
      ([loadedImages, loadedFonts]) => {
        if (!active) return;
        setImages(loadedImages);
        setFonts(loadedFonts);
      },
    );
    return () => {
      active = false;
    };
  }, [sessionId, imageCount]);

  const html = useMemo(() => {
    if (images === null || meta === undefined) return null;
    return buildListingHtml({
      report,
      profile: listing,
      labels: buildListingLabels(listing, meta),
      images,
      fonts,
      classes: buildSpeedClasses(meta.adoption_speed_classes ?? []),
    });
  }, [report, listing, images, fonts, meta]);

  // The download is the preview's own bytes, handed to the browser as a Blob.
  const [href, setHref] = useState<string | null>(null);
  useEffect(() => {
    if (html === null) return;
    const url = URL.createObjectURL(new Blob([html], { type: "text/html" }));
    setHref(url);
    return () => URL.revokeObjectURL(url);
  }, [html]);

  return (
    <section>
      <h2 className="font-[family-name:var(--font-display)] text-2xl">Your listing, rewritten</h2>
      <p className="mt-2 text-sm text-muted">
        A ready-to-use adoption listing built from this report. Download it as a single file, or open
        it and print to PDF.
      </p>

      {html === null ? (
        <div className="mt-6 h-[420px] animate-pulse rounded-xl border border-ink/10 bg-surface" />
      ) : (
        <>
          <div className="mt-6 overflow-hidden rounded-xl border border-ink/10">
            <iframe
              title="Listing preview"
              sandbox=""
              srcDoc={html}
              className="h-[620px] w-full border-0 bg-surface"
            />
          </div>
          {href && (
            <div className="mt-4">
              <a
                href={href}
                download={fileName(listing, sessionId)}
                className="inline-flex items-center justify-center rounded-full px-6 py-3 text-sm font-medium transition bg-transparent text-ink hover:bg-ink/5 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal"
              >
                Download listing
              </a>
            </div>
          )}
        </>
      )}
    </section>
  );
}
