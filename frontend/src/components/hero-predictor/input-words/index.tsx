import { useInputWords } from "@/src/components/hero-predictor/input-words/useInputWords";
import { Input } from "@/src/components/ui/input";
import { Search } from "lucide-react";

export default function InputWords() {
  const {
    userInput,
    setUserInput,
    prediction,
    isLoadingVectorDatabase,
    isLoadingModel,
    isPredicting,
  } = useInputWords();

  return (
    <div className="relative flex items-center">
      <Search className="absolute h-3 w-3 left-3" />
      <Input
        type="text"
        placeholder="Ajoute tes mots..."
        className="pr-2.5 pl-[30px]"
        value={userInput}
        onChange={(e) => setUserInput(e.target.value)}
      />
    </div>
  );
}
