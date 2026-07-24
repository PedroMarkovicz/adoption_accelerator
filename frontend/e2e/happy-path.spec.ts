import { test, expect } from "@playwright/test";

// Requires the FastAPI backend running on :8000 (with OPENAI_API_KEY for the
// full path; without it the verdict still renders and this test still passes).
test("wizard to dossier happy path", async ({ page }) => {
  // The dossier assertion below already waits up to 90s for the multi-agent
  // pipeline; raise the overall test timeout so that budget is actually
  // honored instead of the default 30s cutting the test short.
  test.setTimeout(120000);
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

  // Advance through Health to Listing
  await page.getByRole("button", { name: "Continue" }).click(); // -> Health
  await page.getByRole("button", { name: "Continue" }).click(); // -> Listing

  // Attach the fixture photo while on the listing step (hidden file input).
  await expect(page.getByText("The listing")).toBeVisible();
  await page.setInputFiles('input[type="file"]', "e2e/fixtures/pet.jpg");

  await page.getByRole("button", { name: "Continue" }).click(); // -> Review

  await page.getByRole("button", { name: /predict adoption speed/i }).click();

  // Assembling then Dossier
  await expect(page.getByText(/Assembling the case|The verdict/)).toBeVisible();
  await expect(page.getByText("The verdict")).toBeVisible({ timeout: 90000 });

  // The uploaded photo renders in both the verdict hero and the photo-feedback
  // card (intended duplication with a single photo), so scope to .first().
  await expect(page.getByAltText(/Uploaded photo 1 of 1/).first()).toBeVisible({ timeout: 30000 });
});
