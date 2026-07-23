import { test, expect } from "@playwright/test";

// Requires the FastAPI backend running on :8000 (with OPENAI_API_KEY for the
// full path; without it the verdict still renders and this test still passes).
test("wizard to dossier happy path", async ({ page }) => {
  await page.goto("/predict");
  await expect(page.getByText("The basics")).toBeVisible();

  // Step 1 -> Appearance
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.getByText("Appearance")).toBeVisible();

  // Choose a primary breed (open the Radix select, pick the first real option).
  // The trigger's accessible text is the currently selected item's label
  // ("Mixed / Unknown" by default), not a "Select" placeholder, since the
  // form's default value already matches a real option -- so target the
  // first combobox on the step (Primary breed is listed first) instead.
  await page.getByRole("combobox").first().click();
  await page.getByRole("option").nth(1).click();

  // Advance through Health and Listing to Review
  await page.getByRole("button", { name: "Continue" }).click(); // -> Health
  await page.getByRole("button", { name: "Continue" }).click(); // -> Listing
  await page.getByRole("button", { name: "Continue" }).click(); // -> Review

  await page.getByRole("button", { name: /predict adoption speed/i }).click();

  // Assembling then Dossier
  await expect(page.getByText(/Assembling the case|The verdict/)).toBeVisible();
  await expect(page.getByText("The verdict")).toBeVisible({ timeout: 90000 });
});
