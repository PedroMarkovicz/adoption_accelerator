import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Button } from "@/components/ui/Button";
import { RadioPills } from "@/components/ui/RadioPills";

describe("ui primitives", () => {
  it("Button fires onClick", async () => {
    const onClick = vi.fn();
    render(<Button onClick={onClick}>Continue</Button>);
    await userEvent.click(screen.getByRole("button", { name: "Continue" }));
    expect(onClick).toHaveBeenCalledOnce();
  });

  it("RadioPills selects an option", async () => {
    const onChange = vi.fn();
    render(<RadioPills value="Dog" onChange={onChange} options={["Dog", "Cat"]} ariaLabel="Pet type" />);
    await userEvent.click(screen.getByRole("radio", { name: "Cat" }));
    expect(onChange).toHaveBeenCalledWith("Cat");
  });
});
