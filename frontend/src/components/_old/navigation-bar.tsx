import Logo from "@/public/svgs/logo.svg";
import { DarkmodeToggle } from "@/src/components/_old/darkmode-toggle";
import Link from "next/link";

export function NavigationBar() {
  return (
    <div className="flex items-center justify-between w-full">
      <Link href="/" className="h-full">
        <Logo className="h-10 fill-accent-foreground" />
      </Link>
      <DarkmodeToggle />
    </div>
  );
}
