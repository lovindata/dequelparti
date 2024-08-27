import InputWords from "@/src/components/HeroPredictor/InputWords";
import { SubTitle } from "@/src/components/HeroPredictor/SubTitle";
import { HeroTitle } from "@/src/components/HeroPredictor/Title";

export default function HeroPredictor() {
  return (
    <div className="flex flex-col space-y-3 px-6">
      <HeroTitle />
      <SubTitle />
      <InputWords />
    </div>
  );
}
