import { z } from "zod";

export const petProfileSchema = z.object({
  pet_type: z.enum(["Dog", "Cat"]),
  name: z.string().max(100),
  age_months: z.number().int().min(0).max(255),
  gender: z.enum(["Male", "Female", "Mixed"]),
  breed1: z.number().int().min(1, "Please choose a primary breed"),
  breed2: z.number().int().min(0),
  color1: z.number().int().min(0),
  color2: z.number().int().min(0),
  color3: z.number().int().min(0),
  maturity_size: z.union([z.literal(1), z.literal(2), z.literal(3), z.literal(4)]),
  fur_length: z.union([z.literal(1), z.literal(2), z.literal(3)]),
  vaccinated: z.enum(["Yes", "No", "Not Sure"]),
  dewormed: z.enum(["Yes", "No", "Not Sure"]),
  sterilized: z.enum(["Yes", "No", "Not Sure"]),
  health: z.enum(["Healthy", "Minor Injury", "Serious Injury"]),
  fee: z.number().min(0),
  quantity: z.number().int().min(1),
  state: z.number().int().min(0),
  description: z.string().max(4000),
});

export type PetProfileFormValues = z.infer<typeof petProfileSchema>;

export const defaultValues: PetProfileFormValues = {
  pet_type: "Dog", name: "", age_months: 6, gender: "Male",
  breed1: 0, breed2: 0, color1: 0, color2: 0, color3: 0,
  maturity_size: 2, fur_length: 1,
  vaccinated: "Not Sure", dewormed: "Not Sure", sterilized: "Not Sure", health: "Healthy",
  fee: 0, quantity: 1, state: 0, description: "",
};

export type StepId = "basics" | "appearance" | "health" | "listing";

export const STEP_FIELDS: Record<StepId, (keyof PetProfileFormValues)[]> = {
  basics: ["pet_type", "name", "age_months", "gender", "quantity"],
  appearance: ["breed1", "breed2", "color1", "color2", "color3", "maturity_size", "fur_length"],
  health: ["vaccinated", "dewormed", "sterilized", "health", "fee"],
  listing: ["state", "description"],
};
