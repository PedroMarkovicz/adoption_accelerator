import { cn } from "@/lib/cn";
import { forwardRef } from "react";

type Props = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "ghost";
};

export const Button = forwardRef<HTMLButtonElement, Props>(function Button(
  { className, variant = "primary", ...props }, ref) {
  return (
    <button
      ref={ref}
      className={cn(
        "inline-flex items-center justify-center rounded-full px-6 py-3 text-sm font-medium transition",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal disabled:opacity-50",
        variant === "primary" && "bg-ink text-paper hover:opacity-90",
        variant === "ghost" && "bg-transparent text-ink hover:bg-ink/5",
        className,
      )}
      {...props}
    />
  );
});
