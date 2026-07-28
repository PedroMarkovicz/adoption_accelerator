import { describe, expect, it } from "vitest";
import { buildListingLabels } from "@/lib/listing/labels";
import type { MetaResponse, PetProfileRequest } from "@/lib/types";

const meta = {
  breeds: [
    { id: 307, type: 1, name: "Mixed Breed" },
    { id: 141, type: 1, name: "Beagle" },
  ],
  colors: [
    { id: 1, name: "Black" },
    { id: 2, name: "Brown" },
  ],
  states: [{ id: 41326, name: "Selangor" }],
  maturity_sizes: [{ id: 2, label: "Medium" }],
  fur_lengths: [{ id: 1, label: "Short" }],
} as unknown as MetaResponse;

const base = {
  pet_type: "Dog",
  name: "Milo",
  age_months: 8,
  gender: "Male",
  breed1: 307,
  breed2: 0,
  color1: 1,
  color2: 0,
  color3: 0,
  maturity_size: 2,
  fur_length: 1,
  vaccinated: "Yes",
  dewormed: "Yes",
  sterilized: "No",
  health: "Healthy",
  fee: 50,
  quantity: 1,
  state: 41326,
  video_amt: 0,
  description: "",
} as unknown as PetProfileRequest;

describe("buildListingLabels", () => {
  it("titles the listing with the pet's name", () => {
    expect(buildListingLabels(base, meta).title).toBe("Meet Milo");
  });

  it("falls back to an age band and breed when there is no name", () => {
    const labels = buildListingLabels({ ...base, name: "  " }, meta);
    expect(labels.title).toBe("A young Mixed Breed dog");
  });

  it("falls back to the species alone when the breed does not resolve", () => {
    const labels = buildListingLabels({ ...base, name: "", breed1: 9999 }, meta);
    expect(labels.title).toBe("A young dog");
  });

  it("bands age at the boundaries", () => {
    const title = (age: number) =>
      buildListingLabels({ ...base, name: "", breed1: 9999, age_months: age }, meta).title;
    expect(title(11)).toBe("A young dog");
    expect(title(12)).toBe("An adult dog");
    expect(title(83)).toBe("An adult dog");
    expect(title(84)).toBe("A senior dog");
  });

  it("formats age in years and months", () => {
    expect(buildListingLabels({ ...base, age_months: 1 }, meta).age).toBe("1 month");
    expect(buildListingLabels({ ...base, age_months: 8 }, meta).age).toBe("8 months");
    expect(buildListingLabels({ ...base, age_months: 24 }, meta).age).toBe("2 years");
    expect(buildListingLabels({ ...base, age_months: 27 }, meta).age).toBe("2 years 3 months");
  });

  it("joins a secondary breed", () => {
    const labels = buildListingLabels({ ...base, breed2: 141 }, meta);
    expect(labels.breed).toBe("Mixed Breed / Beagle");
  });

  it("omits unresolved ids instead of rendering undefined", () => {
    const labels = buildListingLabels(
      { ...base, breed1: 9999, color1: 9999, state: 9999 },
      meta,
    );
    expect(labels.breed).toBeNull();
    expect(labels.colors).toBeNull();
    expect(labels.state).toBeNull();
  });

  it("omits every optional field when meta is unavailable", () => {
    const labels = buildListingLabels(base, undefined);
    expect(labels.breed).toBeNull();
    expect(labels.size).toBeNull();
    expect(labels.age).toBe("8 months");
    expect(labels.title).toBe("Meet Milo");
  });

  it("renders a zero fee as Free", () => {
    expect(buildListingLabels({ ...base, fee: 0 }, meta).fee).toBe("Free");
    expect(buildListingLabels({ ...base, fee: 50 }, meta).fee).toBe("RM 50");
  });

  it("lists the health rows in a fixed order", () => {
    expect(buildListingLabels(base, meta).health).toEqual([
      { label: "Vaccinated", value: "Yes" },
      { label: "Dewormed", value: "Yes" },
      { label: "Sterilized", value: "No" },
      { label: "Health", value: "Healthy" },
    ]);
  });

  it("shows the secondary breed alone when the primary does not resolve", () => {
    const labels = buildListingLabels(
      { ...base, breed1: 9999, breed2: 141 },
      meta,
    );
    expect(labels.breed).toBe("Beagle");
  });

  it("deduplicates when the same breed is selected for both slots", () => {
    const labels = buildListingLabels({ ...base, breed1: 141, breed2: 141 }, meta);
    expect(labels.breed).toBe("Beagle");
  });

  it("deduplicates when the same color is selected multiple times", () => {
    const labels = buildListingLabels(
      { ...base, color1: 1, color2: 1, color3: 1 },
      meta,
    );
    expect(labels.colors).toBe("Black");
  });
});
