"use client";
import * as RS from "@radix-ui/react-slider";

export function Slider({ min, max, step = 1, value, onValueChange, ariaLabel }: {
  min: number; max: number; step?: number; value: number; onValueChange: (v: number) => void; ariaLabel: string;
}) {
  return (
    <RS.Root min={min} max={max} step={step} value={[value]} onValueChange={([v]) => onValueChange(v)}
      aria-label={ariaLabel} className="relative flex h-5 w-full touch-none items-center">
      <RS.Track className="relative h-1 grow rounded-full bg-ink/15">
        <RS.Range className="absolute h-full rounded-full bg-teal" />
      </RS.Track>
      <RS.Thumb className="block h-4 w-4 rounded-full border-2 border-teal bg-surface focus-visible:outline-2 focus-visible:outline-teal" />
    </RS.Root>
  );
}
