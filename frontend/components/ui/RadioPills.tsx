"use client";
import * as RadioGroup from "@radix-ui/react-radio-group";
import { cn } from "@/lib/cn";

export function RadioPills({ value, onChange, options, ariaLabel }: {
  value: string; onChange: (v: string) => void; options: string[]; ariaLabel: string;
}) {
  return (
    <RadioGroup.Root value={value} onValueChange={onChange} aria-label={ariaLabel} className="flex flex-wrap gap-2">
      {options.map((opt) => (
        <RadioGroup.Item
          key={opt}
          value={opt}
          className={cn(
            "rounded-full border px-4 py-2 text-sm transition",
            "data-[state=checked]:bg-ink data-[state=checked]:text-paper data-[state=unchecked]:border-ink/20",
            "focus-visible:outline-2 focus-visible:outline-teal",
          )}
        >
          {opt}
        </RadioGroup.Item>
      ))}
    </RadioGroup.Root>
  );
}
