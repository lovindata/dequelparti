import { ChartCard } from "@/src/components/hero-predictor/chart-card";
import { InputText } from "@/src/components/hero-predictor/input-words";
import { SubTitle } from "@/src/components/hero-predictor/sub-title";
import { HeroTitle } from "@/src/components/hero-predictor/title";
import { useHeroPredictor } from "@/src/components/hero-predictor/useHeroPredictor";

export function HeroPredictor() {
  const { userInput, setUserInput, prediction } = useHeroPredictor();

  return (
    <div className="space-y-3 px-6 md:space-y-7 md:px-9">
      <HeroTitle />
      <SubTitle />
      <InputText userInput={userInput} setUserInput={setUserInput} />
      <ChartCard chartData={prediction} />
    </div>
  );
}
