export function reportImageUrl(sessionId: string, index: number): string {
  return `/api/predict/${sessionId}/images/${index}`;
}
