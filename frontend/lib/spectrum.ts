export const SPECTRUM_COLORS = ["#1FA363", "#7DB33A", "#E8B23A", "#E77A3C", "#D14D5A"];

export interface SpeedClass { index: number; label: string; color: string; }

export function buildSpeedClasses(entries: { index: number; label: string }[]): SpeedClass[] {
  return entries.map((e) => ({ ...e, color: SPECTRUM_COLORS[e.index] ?? "#999999" }));
}

export function expectedPosition(probabilities: Record<string, number>): number {
  let num = 0, den = 0;
  for (const [k, v] of Object.entries(probabilities)) {
    num += Number(k) * v;
    den += v;
  }
  return den === 0 ? 0 : num / den;
}
