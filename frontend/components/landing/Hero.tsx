import Link from "next/link";

export function Hero() {
  return (
    <section className="mx-auto max-w-3xl px-6 pt-24 pb-16 text-center">
      <p className="font-mono text-xs uppercase tracking-[0.2em] text-muted">Multimodal ML - generative - agentic</p>
      <h1 className="mt-4 font-[family-name:var(--font-display)] text-5xl leading-tight md:text-6xl">
        See how fast a pet will be adopted, and how to help.
      </h1>
      <p className="mx-auto mt-6 max-w-xl text-lg text-muted">
        A model reads the listing, an agent measures what would speed things up, and a dossier tells you exactly what to change.
      </p>
      <div className="mt-10">
        <Link href="/predict" className="inline-flex items-center justify-center rounded-full bg-ink px-8 py-4 text-paper hover:opacity-90">
          Predict adoption speed
        </Link>
      </div>
    </section>
  );
}
