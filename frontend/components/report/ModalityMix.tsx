"use client";
import type { PredictionEvidence } from "@/lib/types";
import { PieChart, Pie, Cell, ResponsiveContainer, Legend } from "recharts";

const COLORS: Record<string, string> = { tabular: "#0E7C7B", text: "#E8B23A", image: "#E77A3C" };

export function ModalityMix({ prediction }: { prediction: PredictionEvidence }) {
  const data = Object.entries(prediction.modality_contributions).map(([name, value]) => ({ name, value }));
  return (
    <div className="h-56">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie data={data} dataKey="value" nameKey="name" innerRadius={50} outerRadius={80} paddingAngle={2}>
            {data.map((d) => <Cell key={d.name} fill={COLORS[d.name] ?? "#999"} />)}
          </Pie>
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
