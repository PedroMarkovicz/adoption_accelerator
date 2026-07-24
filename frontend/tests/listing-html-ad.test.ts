import { describe, expect, it } from "vitest";
import { buildListingHtml } from "@/lib/listing/buildListingHtml";
import { input, labels, report } from "./fixtures/listing";

describe("buildListingHtml - document and ad page", () => {
  it("produces a complete standalone document", () => {
    const html = buildListingHtml(input());
    expect(html.startsWith("<!doctype html>")).toBe(true);
    expect(html).toContain("<style>");
    expect(html).not.toContain("<script");
  });

  it("renders the title and the profile fields", () => {
    const html = buildListingHtml(input());
    expect(html).toContain("Meet Milo");
    expect(html).toContain("Mixed Breed");
    expect(html).toContain("8 months");
    expect(html).toContain("RM 50");
    expect(html).toContain("Vaccinated");
  });

  it("escapes a pet name that contains markup", () => {
    const html = buildListingHtml(
      input({ labels: { ...labels, title: 'Meet <script>alert("x")</script>' } as never }),
    );
    expect(html).not.toContain("<script>alert");
    expect(html).toContain("&lt;script&gt;");
  });

  it("escapes markup inside the description", () => {
    const html = buildListingHtml(
      input({
        report: { ...report, optimized_description: "<img src=x onerror=alert(1)>" } as never,
      }),
    );
    expect(html).not.toContain("<img src=x");
    expect(html).toContain("&lt;img src=x");
  });

  it("uses best_photo_index as the hero", () => {
    const html = buildListingHtml(input());
    const hero = html.slice(html.indexOf('class="hero"'));
    expect(hero.slice(0, 200)).toContain("base64,BBB");
  });

  it("uses photo 0 as the hero when visual evidence is absent", () => {
    const html = buildListingHtml(input({ report: { ...report, visual: null } as never }));
    const hero = html.slice(html.indexOf('class="hero"'));
    expect(hero.slice(0, 200)).toContain("base64,AAA");
  });

  it("emits no img tag at all when there are no images", () => {
    const html = buildListingHtml(input({ images: [] }));
    expect(html).not.toContain("<img");
    expect(html).toContain("Meet Milo");
  });

  it("falls back to the original description with an explicit note", () => {
    const html = buildListingHtml(
      input({ report: { ...report, optimized_description: null } as never }),
    );
    expect(html).toContain("dog for adoption pls contact");
    expect(html).toContain("the rewrite was unavailable for this run");
  });

  it("omits the description block when both descriptions are empty", () => {
    const html = buildListingHtml(
      input({
        report: { ...report, optimized_description: null } as never,
        profile: { name: "Milo", description: "" } as never,
      }),
    );
    expect(html).not.toContain("the rewrite was unavailable");
    expect(html).toContain("Meet Milo");
  });

  it("embeds fonts only when they are supplied", () => {
    expect(buildListingHtml(input())).not.toContain("@font-face");
    const withFonts = buildListingHtml(
      input({ fonts: { display: "data:font/woff2;base64,DDD", body: "data:font/woff2;base64,BBB" } }),
    );
    expect(withFonts).toContain("@font-face");
    expect(withFonts).toContain("base64,DDD");
  });
});
