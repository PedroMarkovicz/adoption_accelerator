"use client";
import Link from "next/link";

export function SiteHeader() {
  return (
    <header className="border-b border-ink/10 bg-paper">
      <nav aria-label="Main" className="mx-auto flex max-w-5xl items-center justify-between px-4 py-4 sm:px-6">
        <Link
          href="/"
          className="font-[family-name:var(--font-display)] text-lg text-ink focus-visible:outline-2 focus-visible:outline-teal"
        >
          Adoption Accelerator
        </Link>
        <div className="flex items-center gap-6">
          <Link
            href="/predict"
            className="text-sm text-muted hover:text-ink focus-visible:outline-2 focus-visible:outline-teal"
          >
            Predict
          </Link>
          <Link
            href="/explore"
            className="text-sm text-muted hover:text-ink focus-visible:outline-2 focus-visible:outline-teal"
          >
            Explore
          </Link>
        </div>
      </nav>
    </header>
  );
}
