import InputWords from "@/src/components/hero-predictor/input-words";
import { SubTitle } from "@/src/components/hero-predictor/sub-title";
import { HeroTitle } from "@/src/components/hero-predictor/title";

export default function HeroPredictor() {
  return (
    <div className="flex flex-col space-y-3 px-6">
      <HeroTitle />
      <SubTitle />
      <InputWords />
    </div>
  );
}
