"use client";
import { useState } from "react";
import { cn } from "@/lib/cn";
import { reportImageUrl } from "@/lib/images";

export function ReportPhoto({ sessionId, index, total, className }: {
  sessionId: string;
  index: number;
  total: number;
  className?: string;
}) {
  const [failed, setFailed] = useState(false);

  if (failed) {
    return (
      <div className={cn(
        "flex h-full w-full items-center justify-center rounded-lg bg-ink/5 p-4 text-center text-xs text-muted",
        className,
      )}>
        Photo no longer available
      </div>
    );
  }

  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={reportImageUrl(sessionId, index)}
      alt={`Uploaded photo ${index + 1} of ${total}`}
      loading="lazy"
      onError={() => setFailed(true)}
      className={cn("h-full w-full rounded-lg object-cover", className)}
    />
  );
}
