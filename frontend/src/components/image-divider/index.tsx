import imageDivierWebp from "@/src/components/image-divider/image-divider.webp";
import Image from "next/image";

export function ImageDivider() {
  return (
    <Image
      src={imageDivierWebp}
      alt="Assemblée nationale"
      style={{ width: "100%", height: "auto" }} // For responsive images
    />
  );
}
