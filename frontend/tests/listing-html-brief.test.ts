import { describe, expect, it } from "vitest";
import { buildListingHtml } from "@/lib/listing/buildListingHtml";
import { input, recommendations, report } from "./fixtures/listing";

describe("buildListingHtml - evidence brief", () => {
  it("states the prediction and its confidence", () => {
    const html = buildListingHtml(input());
    expect(html).toContain("Why this works");
    expect(html).toContain("Adopted within a week");
    expect(html).toContain("60%");
  });

  it("marks exactly the predicted class on the spectrum", () => {
    const html = buildListingHtml(input());
    expect(html.match(/class="seg marker"/g) ?? []).toHaveLength(1);
    const marked = html.match(/class="seg marker" style="background:([^"]+)"/);
    expect(marked?.[1]).toBe("#7DB33A");
  });

  it("shows the original and the rewritten description side by side", () => {
    const html = buildListingHtml(input());
    expect(html).toContain("Before");
    expect(html).toContain("After");
    expect(html).toContain("dog for adoption pls contact");
  });

  it("collapses before/after to a note when there is no rewrite", () => {
    const html = buildListingHtml(
      input({ report: { ...report, optimized_description: null } as never }),
    );
    expect(html).not.toContain(">After<");
    expect(html).toContain("No rewritten description was produced for this run");
  });

  it("lists at most the top three recommendations, by priority", () => {
    const html = buildListingHtml(
      input({ report: { ...report, recommendations } as never }),
    );
    expect(html).toContain("Add a second clear photo");
    expect(html).toContain("one class faster");
    expect(html).toContain("Lower the fee to RM 20");
    expect(html).not.toContain("Rewrite the title");
  });

  it("orders recommendations by priority regardless of array order", () => {
    const shuffled = { recommendations: [...recommendations.recommendations].reverse() };
    const html = buildListingHtml(input({ report: { ...report, recommendations: shuffled } as never }));
    expect(html.indexOf("Add a second clear photo")).toBeLessThan(html.indexOf("Sterilize before listing"));
    expect(html).not.toContain("Rewrite the title");
  });

  it("omits the actions block entirely when there are no recommendations", () => {
    const html = buildListingHtml(input());
    expect(html).not.toContain("What would move the needle");
  });

  it("escapes markup inside a recommendation", () => {
    const evil = { recommendations: [
      { action: "<b>bold</b>", priority: 1, measured_impact: { expected_speedup: "<i>x</i>" } },
    ] };
    const html = buildListingHtml(input({ report: { ...report, recommendations: evil } as never }));
    expect(html).not.toContain("<b>bold</b>");
    expect(html).toContain("&lt;b&gt;bold&lt;/b&gt;");
  });

  it("prints the provenance footer", () => {
    const html = buildListingHtml(input());
    expect(html).toContain("tuned_v1");
  });
});
