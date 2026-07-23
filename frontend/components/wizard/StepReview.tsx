"use client";
import { useFormContext } from "react-hook-form";
import type { PetProfileFormValues } from "@/lib/schema";
import { Card } from "@/components/ui/Card";

export function StepReview({ images }: { images: File[] }) {
  const { getValues } = useFormContext<PetProfileFormValues>();
  const v = getValues();
  return (
    <Card>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-3 font-mono text-sm">
        <dt className="text-muted">Type</dt><dd>{v.pet_type}</dd>
        <dt className="text-muted">Age</dt><dd>{v.age_months} months</dd>
        <dt className="text-muted">Gender</dt><dd>{v.gender}</dd>
        <dt className="text-muted">Health</dt><dd>{v.health}</dd>
        <dt className="text-muted">Fee</dt><dd>{v.fee} MYR</dd>
        <dt className="text-muted">Photos</dt><dd>{images.length}</dd>
      </dl>
    </Card>
  );
}
