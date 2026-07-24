import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { ListingPreview } from "@/components/report/ListingPreview";

vi.mock("@/lib/listing/assets", () => ({
  loadListingImages: vi.fn(async () => [{ index: 0, dataUri: "data:image/jpeg;base64,AAA" }]),
  loadListingFonts: vi.fn(async () => null),
}));

vi.mock("@/lib/useMeta", () => ({
  useMeta: () => ({
    data: {
      adoption_speed_classes: [
        { index: 0, label: "Same day" },
        { index: 1, label: "Within a week" },
        { index: 2, label: "Within a month" },
        { index: 3, label: "Within three months" },
        { index: 4, label: "Not adopted" },
      ],
      breeds: [{ id: 307, type: 1, name: "Mixed Breed" }],
      colors: [], states: [], maturity_sizes: [], fur_lengths: [],
    },
  }),
}));

import { profile as listing, report } from "./fixtures/listing";

// jsdom has no createObjectURL; capture the Blob so the download can be
// compared byte for byte against what the iframe was given.
const blobs: Blob[] = [];
beforeEach(() => {
  blobs.length = 0;
  vi.stubGlobal("URL", {
    ...URL,
    createObjectURL: (b: Blob) => {
      blobs.push(b);
      return `blob:mock/${blobs.length}`;
    },
    revokeObjectURL: () => {},
  });
});

describe("ListingPreview", () => {
  it("renders the artifact in a fully sandboxed iframe", async () => {
    render(<ListingPreview report={report} listing={listing} />);
    const frame = await screen.findByTitle("Listing preview");
    expect(frame.getAttribute("sandbox")).toBe("");
    await waitFor(() => expect(frame.getAttribute("srcdoc")).toContain("Meet Milo"));
  });

  it("downloads exactly the document it previewed", async () => {
    render(<ListingPreview report={report} listing={listing} />);
    const frame = await screen.findByTitle("Listing preview");
    await waitFor(() => expect(frame.getAttribute("srcdoc")).toBeTruthy());

    const link = (await screen.findByRole("link", { name: /download/i })) as HTMLAnchorElement;
    expect(link.getAttribute("download")).toBe("listing-milo-s1.html");
    expect(blobs).toHaveLength(1);
    expect(await blobs[0].text()).toBe(frame.getAttribute("srcdoc"));
  });

  it("names the file 'pet' when the listing has no name", async () => {
    render(<ListingPreview report={report} listing={{ ...(listing as object), name: "" } as never} />);
    const link = (await screen.findByRole("link", { name: /download/i })) as HTMLAnchorElement;
    expect(link.getAttribute("download")).toBe("listing-pet-s1.html");
  });
});
