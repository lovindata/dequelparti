import { Input } from "@/src/components/shadcn/ui/input";
import { Search } from "lucide-react";

interface Props {
  userInput: string;
  setUserInput: (value: string) => void;
}

export function InputWords({ userInput, setUserInput }: Props) {
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
