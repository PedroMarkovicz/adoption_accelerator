const STEPS = [
  ["Describe the pet", "Tabular details, photos, and a free-text description."],
  ["The board convenes", "A visual analyst reads the photos; a data analyst reads the drivers."],
  ["Impact is measured", "A recommendation agent re-runs the ensemble to measure real speedups."],
  ["Read the dossier", "A clear verdict, actions that work, and the evidence underneath."],
];

export function HowItWorks() {
  return (
    <section className="mx-auto max-w-3xl px-6 py-16">
      <ol className="grid gap-8 md:grid-cols-2">
        {STEPS.map(([title, body], i) => (
          <li key={title}>
            <p className="font-mono text-sm text-teal">{String(i + 1).padStart(2, "0")}</p>
            <h3 className="mt-2 font-[family-name:var(--font-display)] text-xl">{title}</h3>
            <p className="mt-1 text-muted">{body}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}
