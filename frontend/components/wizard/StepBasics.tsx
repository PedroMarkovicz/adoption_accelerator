"use client";
import { Controller, useFormContext } from "react-hook-form";
import type { PetProfileFormValues } from "@/lib/schema";
import { Field } from "@/components/ui/Field";
import { RadioPills } from "@/components/ui/RadioPills";
import { Slider } from "@/components/ui/Slider";

export function StepBasics() {
  const { control, register, watch, formState: { errors } } = useFormContext<PetProfileFormValues>();
  const age = watch("age_months");
  const quantity = watch("quantity");
  return (
    <div className="flex flex-col gap-6">
      <Field label="Is it a dog or a cat?">
        <Controller control={control} name="pet_type"
          render={({ field }) => <RadioPills value={field.value} onChange={field.onChange} options={["Dog", "Cat"]} ariaLabel="Pet type" />} />
      </Field>
      <Field label="Name (optional)" htmlFor="name">
        <input id="name" {...register("name")} className="rounded-lg border border-ink/15 bg-surface px-3 py-2.5 text-sm" />
      </Field>
      <Field label={`Age: ${age} months`}>
        <Controller control={control} name="age_months"
          render={({ field }) => <Slider min={0} max={120} value={field.value} onValueChange={field.onChange} ariaLabel="Age in months" />} />
      </Field>
      <Field label="Gender">
        <Controller control={control} name="gender"
          render={({ field }) => <RadioPills value={field.value} onChange={field.onChange} options={["Male", "Female", "Mixed"]} ariaLabel="Gender" />} />
      </Field>
      <Field label={`How many pets in this listing? ${quantity} pet${quantity === 1 ? "" : "s"}`} error={errors.quantity?.message}>
        <Controller control={control} name="quantity"
          render={({ field }) => <Slider min={1} max={20} value={field.value} onValueChange={field.onChange} ariaLabel="Quantity" />} />
      </Field>
    </div>
  );
}
