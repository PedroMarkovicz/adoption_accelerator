"use client";
import * as RS from "@radix-ui/react-select";
import { cn } from "@/lib/cn";

export type Option = { id: number | string; label: string };

export function Select({ value, onValueChange, options, placeholder, id }: {
  value?: string; onValueChange: (v: string) => void; options: Option[]; placeholder?: string; id?: string;
}) {
  return (
    <RS.Root value={value} onValueChange={onValueChange}>
      <RS.Trigger id={id} className={cn(
        "inline-flex w-full items-center justify-between rounded-lg border border-ink/15 bg-surface px-3 py-2.5 text-sm",
        "focus-visible:outline-2 focus-visible:outline-teal")}>
        <RS.Value placeholder={placeholder} />
      </RS.Trigger>
      <RS.Portal>
        <RS.Content className="z-50 max-h-72 overflow-auto rounded-lg border border-ink/10 bg-surface shadow-lg">
          <RS.Viewport className="p-1">
            {options.map((o) => (
              <RS.Item key={o.id} value={String(o.id)}
                className="cursor-pointer rounded px-3 py-2 text-sm data-[highlighted]:bg-ink/5 data-[highlighted]:outline-none">
                <RS.ItemText>{o.label}</RS.ItemText>
              </RS.Item>
            ))}
          </RS.Viewport>
        </RS.Content>
      </RS.Portal>
    </RS.Root>
  );
}
