import { Button } from "@/components/ui/Button";
export function ReportError({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="mx-auto max-w-xl px-6 py-24 text-center">
      <h1 className="font-[family-name:var(--font-display)] text-2xl">The run could not finish</h1>
      <p className="mt-3 text-muted">{message}</p>
      <div className="mt-6"><Button onClick={onRetry}>Try again</Button></div>
    </div>
  );
}
