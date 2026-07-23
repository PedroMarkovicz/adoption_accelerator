export function Field({ label, htmlFor, error, children }: {
  label: string; htmlFor?: string; error?: string; children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={htmlFor} className="text-sm font-medium text-ink">{label}</label>
      {children}
      {error && <p role="alert" className="text-sm text-[var(--spectrum-4)]">{error}</p>}
    </div>
  );
}
