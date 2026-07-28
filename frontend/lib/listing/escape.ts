const ENTITIES: Record<string, string> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

/** Escape a value for interpolation into HTML text or an attribute. */
export function escapeHtml(value: string): string {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ENTITIES[c]);
}
