import nextConfig from "@/next.config.mjs";
import ens from "@/src/components/programs-carousel/ProgramImage/assets/ens.webp";
import nfp from "@/src/components/programs-carousel/ProgramImage/assets/nfp.webp";
import rn from "@/src/components/programs-carousel/ProgramImage/assets/rn.webp";
import Image, { StaticImageData } from "next/image";
import { useMemo } from "react";

interface Props {
  type: "Ensemble" | "Nouveau Front Populaire" | "Rassemblement National";
}

export function ProgramImage({ type }: Props) {
  const programs: Record<
    Props["type"],
    { image: StaticImageData; href: string }
  > = useMemo(
    () => ({
      Ensemble: { image: ens, href: `${nextConfig.basePath}/prgms/ENS.pdf` },
      "Nouveau Front Populaire": {
        image: nfp,
        href: `${nextConfig.basePath}/prgms/NFP.pdf`,
      },
      "Rassemblement National": {
        image: rn,
        href: `${nextConfig.basePath}/prgms/RN.pdf`,
      },
    }),
    [],
  );

  return (
    <div className="flex justify-center">
      <a href={programs[type].href} download>
        <Image
          src={programs[type].image}
          alt={type}
          height={320}
          width={0}
          className="rounded-xl border shadow"
        />
      </a>
    </div>
  );
}
