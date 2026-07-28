"use client";
import { Controller, useFormContext } from "react-hook-form";
import type { PetProfileFormValues } from "@/lib/schema";
import { useMeta } from "@/lib/useMeta";
import { Field } from "@/components/ui/Field";
import { Select, type Option } from "@/components/ui/Select";

export function StepAppearance() {
  const { control, watch, formState: { errors } } = useFormContext<PetProfileFormValues>();
  const { data: meta } = useMeta();
  const petType = watch("pet_type");
  const typeId = petType === "Dog" ? 1 : 2;
  const breedOpts: Option[] = [{ id: 0, label: "Mixed / Unknown" },
    ...(meta?.breeds.filter((b) => b.type === typeId).map((b) => ({ id: b.id, label: b.name })) ?? [])];
  const colorOpts: Option[] = [{ id: 0, label: "Not specified" },
    ...(meta?.colors.map((c) => ({ id: c.id, label: c.name })) ?? [])];
  const sizeOpts: Option[] = meta?.maturity_sizes.map((s) => ({ id: s.id, label: s.label })) ?? [];
  const furOpts: Option[] = meta?.fur_lengths.map((f) => ({ id: f.id, label: f.label })) ?? [];

  const numField = (name: keyof PetProfileFormValues, label: string, options: Option[], error?: string) => (
    <Field label={label} error={error}>
      <Controller control={control} name={name as never}
        render={({ field }) => (
          <Select value={String(field.value)} onValueChange={(v) => field.onChange(Number(v))} options={options} placeholder="Select" />
        )} />
    </Field>
  );

  return (
    <div className="flex flex-col gap-6">
      {numField("breed1", "Primary breed", breedOpts, errors.breed1?.message)}
      {numField("breed2", "Secondary breed (optional)", breedOpts)}
      {numField("color1", "Primary color", colorOpts)}
      {numField("color2", "Secondary color (optional)", colorOpts)}
      {numField("color3", "Third color (optional)", colorOpts)}
      {numField("maturity_size", "Size when full-grown", sizeOpts)}
      {numField("fur_length", "Fur length", furOpts)}
    </div>
  );
}
