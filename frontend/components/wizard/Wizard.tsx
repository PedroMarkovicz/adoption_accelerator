"use client";
import { useState } from "react";
import { useForm, FormProvider } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter } from "next/navigation";
import { petProfileSchema, defaultValues, STEP_FIELDS, type PetProfileFormValues, type StepId } from "@/lib/schema";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Progress } from "@/components/ui/Progress";
import { StepBasics } from "./StepBasics";
import { StepAppearance } from "./StepAppearance";
import { StepHealth } from "./StepHealth";
import { StepListing } from "./StepListing";
import { StepReview } from "./StepReview";

const STEPS: { id: StepId | "review"; title: string }[] = [
  { id: "basics", title: "The basics" },
  { id: "appearance", title: "Appearance" },
  { id: "health", title: "Health & care" },
  { id: "listing", title: "The listing" },
  { id: "review", title: "Review" },
];

export function Wizard() {
  const router = useRouter();
  const methods = useForm<PetProfileFormValues>({ resolver: zodResolver(petProfileSchema), defaultValues, mode: "onTouched" });
  const [stepIndex, setStepIndex] = useState(0);
  const [images, setImages] = useState<File[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const step = STEPS[stepIndex];

  async function next() {
    if (step.id !== "review") {
      const ok = await methods.trigger(STEP_FIELDS[step.id]);
      if (!ok) return;
    }
    setStepIndex((i) => Math.min(i + 1, STEPS.length - 1));
  }
  function back() { setStepIndex((i) => Math.max(i - 1, 0)); }

  async function submit() {
    setSubmitting(true); setSubmitError(null);
    try {
      const values = methods.getValues();
      const form = new FormData();
      form.append("profile", JSON.stringify(values));
      images.forEach((file) => form.append("images", file));
      const res = await api.createPrediction(form);
      router.push(`/report/${res.session_id}`);
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : "Something went wrong. Please try again.");
      setSubmitting(false);
    }
  }

  return (
    <FormProvider {...methods}>
      <div className="mx-auto max-w-2xl px-6 py-12">
        <Progress value={((stepIndex + 1) / STEPS.length) * 100} />
        <p className="mt-3 font-mono text-xs text-muted">Step {stepIndex + 1} / {STEPS.length}</p>
        <h1 className="mt-2 font-[family-name:var(--font-display)] text-3xl">{step.title}</h1>

        <div className="mt-8">
          {step.id === "basics" && <StepBasics />}
          {step.id === "appearance" && <StepAppearance />}
          {step.id === "health" && <StepHealth />}
          {step.id === "listing" && <StepListing images={images} setImages={setImages} />}
          {step.id === "review" && <StepReview images={images} />}
        </div>

        {submitError && <p role="alert" className="mt-4 text-sm text-[var(--spectrum-4)]">{submitError}</p>}

        <div className="mt-10 flex justify-between">
          <Button variant="ghost" onClick={back} disabled={stepIndex === 0}>Back</Button>
          {step.id === "review"
            ? <Button onClick={submit} disabled={submitting}>{submitting ? "Assembling..." : "Predict adoption speed"}</Button>
            : <Button onClick={next}>Continue</Button>}
        </div>
      </div>
    </FormProvider>
  );
}
