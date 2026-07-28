"use client";
import { Controller, useFormContext } from "react-hook-form";
import type { PetProfileFormValues } from "@/lib/schema";
import { Field } from "@/components/ui/Field";
import { RadioPills } from "@/components/ui/RadioPills";
import { Slider } from "@/components/ui/Slider";

const YN = ["Yes", "No", "Not Sure"];

export function StepHealth() {
  const { control, watch } = useFormContext<PetProfileFormValues>();
  const fee = watch("fee");
  const triState = (name: keyof PetProfileFormValues, label: string) => (
    <Field label={label}>
      <Controller control={control} name={name as never}
        render={({ field }) => <RadioPills value={field.value as string} onChange={field.onChange} options={YN} ariaLabel={label} />} />
    </Field>
  );
  return (
    <div className="flex flex-col gap-6">
      {triState("vaccinated", "Vaccinated?")}
      {triState("dewormed", "Dewormed?")}
      {triState("sterilized", "Sterilized?")}
      <Field label="Health status">
        <Controller control={control} name="health"
          render={({ field }) => <RadioPills value={field.value} onChange={field.onChange} options={["Healthy", "Minor Injury", "Serious Injury"]} ariaLabel="Health" />} />
      </Field>
      <Field label={`Adoption fee: ${fee} MYR`}>
        <Controller control={control} name="fee"
          render={({ field }) => <Slider min={0} max={500} step={10} value={field.value} onValueChange={field.onChange} ariaLabel="Adoption fee" />} />
      </Field>
    </div>
  );
}
