import { describe, it, expect } from "vitest";
import { petProfileSchema, defaultValues } from "@/lib/schema";

describe("petProfileSchema", () => {
  it("accepts valid defaults with required fields set", () => {
    const parsed = petProfileSchema.safeParse({ ...defaultValues, breed1: 307, age_months: 12 });
    expect(parsed.success).toBe(true);
  });

  it("rejects age out of range", () => {
    const parsed = petProfileSchema.safeParse({ ...defaultValues, breed1: 307, age_months: 999 });
    expect(parsed.success).toBe(false);
  });

  it("requires a primary breed", () => {
    const parsed = petProfileSchema.safeParse({ ...defaultValues, breed1: 0, age_months: 12 });
    expect(parsed.success).toBe(false);
  });
});
