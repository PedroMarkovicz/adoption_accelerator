"use client";
import { Controller, useFormContext } from "react-hook-form";
import type { PetProfileFormValues } from "@/lib/schema";
import { useMeta } from "@/lib/useMeta";
import { Field } from "@/components/ui/Field";
import { Select, type Option } from "@/components/ui/Select";
import { PhotoDropzone } from "./PhotoDropzone";

export function StepListing({ images, setImages }: { images: File[]; setImages: (f: File[]) => void }) {
  const { control, register } = useFormContext<PetProfileFormValues>();
  const { data: meta } = useMeta();
  const stateOpts: Option[] = [{ id: 0, label: "Not specified" },
    ...(meta?.states.map((s) => ({ id: s.id, label: s.name })) ?? [])];
  return (
    <div className="flex flex-col gap-6">
      <Field label="Photos">
        <PhotoDropzone images={images} setImages={setImages} />
      </Field>
      <Field label="Description">
        <textarea {...register("description")} rows={6}
          placeholder="Tell adopters about this pet's personality, story, and needs."
          className="rounded-lg border border-ink/15 bg-surface px-3 py-2.5 text-sm" />
      </Field>
      <Field label="State (Malaysia)">
        <Controller control={control} name="state"
          render={({ field }) => <Select value={String(field.value)} onValueChange={(v) => field.onChange(Number(v))} options={stateOpts} placeholder="Select a state" />} />
      </Field>
    </div>
  );
}
