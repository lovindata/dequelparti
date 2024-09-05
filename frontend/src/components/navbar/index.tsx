import nextConfig from "@/next.config.mjs";
import { DarkmodeToggle } from "@/src/components/navbar/darkmode-toggle";
import { cn } from "@/src/lib/utils";
import Image from "next/image";
import Link from "next/link";

interface Props extends React.HTMLAttributes<HTMLDivElement> {}

export function Navbar({ className }: Props) {
  return (
    <div
      className={cn(
        "m-2 flex items-center justify-between rounded-2xl border bg-card px-3 py-2 md:m-4",
        className,
      )}
    >
      <Link
        className="flex items-center space-x-2 font-semibold text-muted-foreground hover:text-current"
        href="/"
      >
        <Image
          src={`${nextConfig.basePath}/imgs/logo.webp`}
          alt={"DeQuelParti"}
          width={62}
          height={32}
        />
        <span>DeQuelParti</span>
      </Link>
      <DarkmodeToggle />
    </div>
  );
}
