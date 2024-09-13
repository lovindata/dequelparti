import nextConfig from "@/next.config.mjs";
import { DarkmodeToggle } from "@/src/components/navbar/darkmode-toggle";
import Image from "next/image";
import Link from "next/link";

export function Navbar() {
  return (
    <nav className="sticky top-0 z-50 p-2 md:p-4 lg:py-4">
      <div className="flex items-center justify-between rounded-2xl border px-3 py-2 backdrop-blur-3xl lg:mx-auto lg:max-w-5xl">
        <Link
          className="flex items-center space-x-2 font-semibold text-muted-foreground hover:text-current"
          href="/"
        >
          <Image
            src={`${nextConfig.basePath}/imgs/logo.webp`}
            alt="DeQuelParti"
            width={62}
            height={32}
          />
          <span>DeQuelParti</span>
        </Link>
        <DarkmodeToggle />
      </div>
    </nav>
  );
}
