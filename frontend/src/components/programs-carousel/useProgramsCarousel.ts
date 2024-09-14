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
  const [carouselLength, setCarouselLength] = useState(0);
  const [carouselCurrentIndex, setCarouselCurrentIndex] = useState(0);

  useEffect(() => {
    if (carouselApi === undefined) return;
    // Initialization
    setCarouselLength(carouselApi.scrollSnapList().length);
    setCarouselCurrentIndex(carouselApi.selectedScrollSnap());
    // Set embla events
    carouselApi.on("resize", () => {
      setCarouselLength(carouselApi.scrollSnapList().length);
      setCarouselCurrentIndex(carouselApi.selectedScrollSnap());
    });
    carouselApi.on("select", () =>
      setCarouselCurrentIndex(carouselApi.selectedScrollSnap()),
    );
  }, [carouselApi]);

  const carouselIndicators = Array.from(
    { length: carouselLength },
    (_, i) => i,
  ).map((i) => i == carouselCurrentIndex);

  return {
    carouselAutoplay,
    setCarouselApi,
    carouselIndicators,
  };
}
