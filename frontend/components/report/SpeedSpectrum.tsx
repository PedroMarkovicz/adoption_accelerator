"use client";
import { motion, useReducedMotion } from "framer-motion";
import type { SpeedClass } from "@/lib/spectrum";
import { expectedPosition } from "@/lib/spectrum";

export function SpeedSpectrum({ classes, markerClass, probabilities, confidence }: {
  classes: SpeedClass[];
  markerClass: number;
  probabilities?: Record<string, number>;
  confidence?: number;
}) {
  const n = classes.length;
  const pos = probabilities ? expectedPosition(probabilities) : markerClass;
  const leftPct = (pos / (n - 1)) * 100;
  // "left" is a plain CSS position property, not one of the transform values
  // MotionConfig's reducedMotion="user" covers, so disable the spring here too.
  const reduceMotion = useReducedMotion();

  return (
    <div className="w-full">
      <div className="relative h-3 w-full overflow-hidden rounded-full"
        style={{ background: `linear-gradient(90deg, ${classes.map((c) => c.color).join(",")})` }}>
      </div>
      <motion.div
        data-testid="spectrum-marker"
        data-class={markerClass}
        className="relative -mt-4 h-5 w-5 rounded-full border-2 border-ink bg-surface"
        initial={{ left: reduceMotion ? `${leftPct}%` : "50%", opacity: reduceMotion ? 1 : 0 }}
        animate={{ left: `${leftPct}%`, opacity: 1 }}
        transition={reduceMotion ? { duration: 0 } : { type: "spring", stiffness: 120, damping: 18 }}
        style={{ transform: "translateX(-50%)" }}
      />
      <div className="mt-3 flex justify-between text-xs text-muted">
        {classes.map((c) => (
          <span key={c.index} className="max-w-[18%] text-center leading-tight">{c.label}</span>
        ))}
      </div>
      {confidence != null && (
        <p className="mt-2 font-mono text-sm text-muted">
          {Math.round(confidence * 100)}% confidence
        </p>
      )}
    </div>
  );
}
