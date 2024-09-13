import { CarouselApi } from "@/src/components/shadcn/ui/carousel";
import Autoplay from "embla-carousel-autoplay";
import { useEffect, useRef, useState } from "react";

export function useProgramsCarousel() {
  const carouselAutoplay = useRef(
    Autoplay({
      delay: 5000, // In milliseconds
    }),
  );
  const [carouselApi, setCarouselApi] = useState<CarouselApi>();
  const [carouselCurrentIndex, setCarouselCurrentIndex] = useState(0);
  const [carouselLength, setCarouselLength] = useState(0);

  useEffect(() => {
    if (carouselApi === undefined) return;
    setCarouselLength(carouselApi.scrollSnapList().length);
    setCarouselCurrentIndex(carouselApi.selectedScrollSnap() + 1);
    carouselApi.on("select", () =>
      setCarouselCurrentIndex(carouselApi.selectedScrollSnap() + 1),
    );
  }, [carouselApi]);

  const carouselIndicators = Array.from(
    { length: carouselLength },
    (_, i) => i + 1,
  ).map((i) => i == carouselCurrentIndex);

  return {
    carouselAutoplay,
    setCarouselApi,
    carouselIndicators,
  };
}
