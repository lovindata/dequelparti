import { Progress } from "@/src/components/shadcn/ui/progress";
import Image from "next/image";

interface Props {
  src:
    | "/dequelparti/imgs/pgrm_ensemble_page-0001.jpg"
    | "/dequelparti/imgs/prgm_nouveau_front_populaire_page-0001.jpg"
    | "/dequelparti/imgs/prgm_rassemblement_national_page-0001.jpg";
  value: number;
}

export function ProgressBar({ src, value }: Props) {
  return (
    <div className="flex flex-col items-center justify-between w-full space-y-3">
      <div className="flex justify-between items-end w-full">
        <Image
          src={src}
          width={60}
          height={60}
          alt={src}
          className="rounded-md"
        />
        <p className="font-black text-primary translate-y-2 text-base">
          {value}%
        </p>
      </div>
      <Progress value={value} />
    </div>
  );
}
