import { cn } from "@/lib/cn";
export function Card({ className, children, "data-testid": dataTestId }: {
  className?: string;
  children: React.ReactNode;
  "data-testid"?: string;
}) {
  return (
    <div data-testid={dataTestId} className={cn("rounded-2xl border border-ink/8 bg-surface p-6", className)}>
      {children}
    </div>
  );
}
