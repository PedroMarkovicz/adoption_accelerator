"use client";
import { useState } from "react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";

export function RewrittenCopy({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <section>
      <h2 className="font-[family-name:var(--font-display)] text-2xl">A stronger description</h2>
      <Card className="mt-6">
        <p className="whitespace-pre-wrap text-[15px] leading-relaxed">{text}</p>
        <div className="mt-4">
          <Button variant="ghost" onClick={() => { navigator.clipboard.writeText(text); setCopied(true); }}>
            {copied ? "Copied" : "Copy description"}
          </Button>
        </div>
      </Card>
    </section>
  );
}
