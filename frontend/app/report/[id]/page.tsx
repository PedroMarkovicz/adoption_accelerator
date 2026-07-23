"use client";
import { use } from "react";
import { useRouter } from "next/navigation";
import { useReportStatus } from "@/lib/useReportStatus";
import { Assembling } from "@/components/report/Assembling";
import { ReportError } from "@/components/report/ReportError";
import { Dossier } from "@/components/report/Dossier";

export default function ReportPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const router = useRouter();
  const { status, report, error } = useReportStatus(id);

  if (status === "error") return <ReportError message={error ?? "Unknown error"} onRetry={() => router.push("/predict")} />;
  if (status !== "done" || !report) return <Assembling />;
  return <Dossier report={report} />;
}
