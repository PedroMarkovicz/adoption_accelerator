"use client";
import { motion, useReducedMotion } from "framer-motion";

const STAGES = [
  "Running the ensemble prediction",
  "Visual analyst reading the photos",
  "Data analyst interpreting the drivers",
  "Recommendation agent measuring impact",
  "Synthesizing the dossier",
];

export function Assembling() {
  // MotionConfig's reducedMotion="user" only neutralizes transform values
  // (x, y, scale, rotate, ...); this opacity pulse needs an explicit check.
  const reduceMotion = useReducedMotion();
  return (
    <div className="mx-auto max-w-xl px-6 py-24">
      <h1 className="font-[family-name:var(--font-display)] text-3xl">Assembling the case</h1>
      <ul className="mt-8 flex flex-col gap-3">
        {STAGES.map((s, i) => (
          <motion.li key={s} initial={{ opacity: reduceMotion ? 1 : 0.3 }} animate={{ opacity: 1 }}
            transition={reduceMotion ? { duration: 0 } : { delay: i * 0.6, repeat: Infinity, repeatType: "reverse", duration: 0.9 }}
            className="flex items-center gap-3 font-mono text-sm text-muted">
            <span className="h-2 w-2 rounded-full bg-teal" /> {s}
          </motion.li>
        ))}
      </ul>
    </div>
  );
}
