"use client";
import { useState } from "react";
import type { AdoptionReport } from "@/lib/types";
import type { SpeedClass } from "@/lib/spectrum";
import { TabsRoot, TabsList, TabsTrigger, TabsContent } from "@/components/ui/Tabs";
import { Card } from "@/components/ui/Card";
import { KeyDrivers } from "./KeyDrivers";
import { ModalityMix } from "./ModalityMix";
import { Probabilities } from "./Probabilities";
import { AgentTrace } from "./AgentTrace";
import { Uncertainty } from "./Uncertainty";

export function PeelBack({ report, classes }: { report: AdoptionReport; classes: SpeedClass[] }) {
  const [open, setOpen] = useState(false);
  return (
    <section className="border-t border-ink/10 pt-8">
      <button onClick={() => setOpen((o) => !o)}
        className="font-mono text-sm text-teal focus-visible:outline-2 focus-visible:outline-teal">
        {open ? "Hide the evidence" : "See how the AI decided"}
      </button>
      {open && (
        <Card className="mt-6">
          <TabsRoot defaultValue="drivers">
            <TabsList className="flex flex-wrap gap-2 border-b border-ink/10 pb-3">
              {[["drivers", "Key drivers"], ["modality", "Modality mix"], ["prob", "Probabilities"], ["trace", "Agent trace"], ["uncertainty", "Uncertainty"]].map(([v, l]) => (
                <TabsTrigger key={v} value={v}
                  className="rounded-full px-3 py-1.5 text-sm data-[state=active]:bg-ink data-[state=active]:text-paper">{l}</TabsTrigger>
              ))}
            </TabsList>
            <div className="pt-6">
              <TabsContent value="drivers"><KeyDrivers prediction={report.prediction} /></TabsContent>
              <TabsContent value="modality"><ModalityMix prediction={report.prediction} /></TabsContent>
              <TabsContent value="prob"><Probabilities prediction={report.prediction} classes={classes} /></TabsContent>
              <TabsContent value="trace"><AgentTrace report={report} /></TabsContent>
              <TabsContent value="uncertainty"><Uncertainty prediction={report.prediction} /></TabsContent>
            </div>
          </TabsRoot>
        </Card>
      )}
    </section>
  );
}
