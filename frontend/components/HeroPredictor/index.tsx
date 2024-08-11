import InputWords from "@/components/HeroPredictor/InputWords";
import { SubTitle } from "@/components/HeroPredictor/SubTitle";
import { HeroTitle } from "@/components/HeroPredictor/Title";

export default function HeroPredictor() {
  return (
    <div className="flex flex-col space-y-3 px-6">
      <HeroTitle />
      <SubTitle />
      <InputWords />
    </div>
  );
}
