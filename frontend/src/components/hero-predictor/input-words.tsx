import { Input } from "@/src/components/shadcn/ui/input";
import { Search } from "lucide-react";

interface Props {
  userInput: string;
  setUserInput: (value: string) => void;
}

export function InputText({ userInput, setUserInput }: Props) {
  return (
    <div className="relative flex items-center">
      <Search className="absolute left-3 h-3 w-3" />
      <Input
        type="text"
        placeholder="Quelque chose"
        className="h-11 pl-[30px] pr-2.5"
        value={userInput}
        onChange={(e) => setUserInput(e.target.value)}
      />
    </div>
  );
}
