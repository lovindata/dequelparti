import { ProgramImage } from "@/src/components/programs-carousel/ProgramImage";
import { useProgramsCarousel } from "@/src/components/programs-carousel/useProgramsCarousel";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
} from "@/src/components/shadcn/ui/carousel";
import { Section } from "@/src/components/shared/atoms/section";
import { cn } from "@/src/lib/utils";

export function ProgramsCarousel() {
  const { carouselAutoplay, setCarouselApi, carouselIndicators } =
    useProgramsCarousel();

  return (
    <Section
      title="Les programmes politiques"
      subTitle="Click pour télécharger et te faire ton propre avis !"
    >
      <Carousel
        plugins={[carouselAutoplay.current]}
        onMouseEnter={carouselAutoplay.current.stop}
        onMouseLeave={carouselAutoplay.current.reset}
        setApi={setCarouselApi}
      >
        <CarouselContent>
          <CarouselItem className="md:basis-1/3">
            <ProgramImage type="Ensemble" />
          </CarouselItem>
          <CarouselItem className="md:basis-1/3">
            <ProgramImage type="Nouveau Front Populaire" />
          </CarouselItem>
          <CarouselItem className="md:basis-1/3">
            <ProgramImage type="Rassemblement National" />
          </CarouselItem>
        </CarouselContent>
      </Carousel>
      <div className="flex flex-row justify-center space-x-2 py-4 text-center text-sm text-muted-foreground md:hidden">
        {carouselIndicators.map((indicator, key) => (
          <span
            key={key}
            className={cn(
              "h-2 w-2 rounded-full",
              indicator === true ? "bg-foreground" : "bg-muted-foreground",
            )}
          />
        ))}
      </div>
    </Section>
  );
}
