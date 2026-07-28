"use client";
import * as RP from "@radix-ui/react-progress";

export function Progress({ value }: { value: number }) {
  return (
    <RP.Root value={value} className="h-1.5 w-full overflow-hidden rounded-full bg-ink/10">
      <RP.Indicator className="h-full bg-teal transition-transform" style={{ transform: `translateX(-${100 - value}%)` }} />
    </RP.Root>
  );
}
