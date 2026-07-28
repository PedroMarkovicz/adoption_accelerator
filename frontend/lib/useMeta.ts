"use client";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";

export function useMeta() {
  return useQuery({ queryKey: ["meta"], queryFn: api.getMeta, staleTime: Infinity });
}
