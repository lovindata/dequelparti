import nextConfig from "@/next.config.mjs";
import Image from "next/image";
import Link from "next/link";
import {
  FaDocker,
  FaGithub,
  FaLinkedin,
  FaPython,
  FaReddit,
  FaMedium,
  FaXTwitter,
} from "react-icons/fa6";

export function Footer() {
  return (
    <footer className="flex flex-col items-center space-y-7 border-t bg-card px-4 py-11 text-sm">
      <Link
        className="flex items-center space-x-2 font-semibold text-muted-foreground hover:text-current"
        href="/"
      >
        <Image
          src={`${nextConfig.basePath}/imgs/logo.webp`}
          alt={"DeQuelParti"}
          width={35}
          height={18}
        />
        <span>DeQuelParti</span>
      </Link>
      <div className="flex space-x-3">
        <Link
          href="https://github.com/lovindata"
          target="_blank"
          rel="noopener noreferrer"
        >
          <FaGithub
            size={18}
            className="fill-muted-foreground hover:fill-current"
          />
        </Link>
        <Link
          href="https://pypi.org/user/jamesjg/"
          target="_blank"
          rel="noopener noreferrer"
        >
          <FaPython
            size={18}
            className="fill-muted-foreground hover:fill-current"
          />
        </Link>
        <Link
          href="https://hub.docker.com/repositories/lovindata"
          target="_blank"
          rel="noopener noreferrer"
        >
          <FaDocker
            size={18}
            className="fill-muted-foreground hover:fill-current"
          />
        </Link>
        <Link
          href="https://twitter.com/jamesjg_"
          target="_blank"
          rel="noopener noreferrer"
        >
          <FaXTwitter
            size={18}
            className="fill-muted-foreground hover:fill-current"
          />
        </Link>
        <Link
          href="https://www.linkedin.com/in/james-jiang-87306b155/"
          target="_blank"
          rel="noopener noreferrer"
        >
          <FaLinkedin
            size={18}
            className="fill-muted-foreground hover:fill-current"
          />
        </Link>
        <Link
          href="https://www.reddit.com/user/lovindata/"
          target="_blank"
          rel="noopener noreferrer"
        >
          <FaReddit
            size={18}
            className="fill-muted-foreground hover:fill-current"
          />
        </Link>
        <Link
          href="https://medium.com/@jamesjg"
          target="_blank"
          rel="noopener noreferrer"
        >
          <FaMedium
            size={18}
            className="fill-muted-foreground hover:fill-current"
          />
        </Link>
      </div>

      <p className="font-semibold text-muted-foreground">Copyright © 2024</p>
      <p className="flex space-x-1 text-xs">
        <span className="font-light text-muted-foreground">Crafted by</span>
        <span>James</span>
      </p>
    </footer>
  );
}
