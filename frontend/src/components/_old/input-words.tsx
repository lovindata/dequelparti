import { Badge } from "@/src/components/ui/badge";
import { Button } from "@/src/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandItem,
  CommandInput,
  CommandList,
  CommandGroup,
} from "@/src/components/ui/command";
// import vocabulary from "@/data/vocabulary.json";
import Fuse from "fuse.js";
import React from "react";

interface Props {
  words: string[];
  appendWord: (word: string) => void;
  popWordAt: (idx: number) => void;
  popAllWords: () => void;
}

export function InputWords({
  words,
  appendWord,
  popWordAt,
  popAllWords,
}: Props) {
  const [inputWord, setInputWord] = React.useState("");
  // let vocabularyShown =
  //   inputWord === ""
  //     ? []
  //     : new Fuse(Object.keys(vocabulary))
  //         .search(inputWord)
  //         .map((_) => _.item)
  //         .filter((_) => !words.includes(_))
  //         .slice(0, 3);
  let vocabularyShown: string[] = [];

  return (
    <div className="space-y-6">
      <div className="text-3xl text-accent-foreground text-center flex flex-col space-y-1">
        <p>Découvre à quel parti politique tes mots correspondent!</p>
      </div>
      <div className="space-y-4">
        <div className="flex justify-between space-x-2 w-full">
          <Command className="relative">
            <CommandInput
              value={inputWord}
              placeholder="Rentre tes mots!"
              onValueChange={(value) => setInputWord(value)}
            />
            <CommandList>
              {inputWord !== "" && (
                <CommandEmpty>Pas de mots trouvés.</CommandEmpty>
              )}
              {inputWord !== "" && (
                <CommandGroup>
                  {vocabularyShown.map((word) => (
                    <CommandItem
                      key={word}
                      onSelect={() => {
                        appendWord(word);
                        setInputWord("");
                      }}
                      className="cursor-pointer"
                    >
                      {word}
                    </CommandItem>
                  ))}
                </CommandGroup>
              )}
            </CommandList>
          </Command>
          <Button variant="secondary" onClick={() => popAllWords()}>
            Reset
          </Button>
        </div>
        <div className="flex-wrap flex justify-center items-center gap-2">
          {words.map((word, idx) => (
            <Badge
              className="cursor-pointer"
              key={idx}
              onClick={() => popWordAt(idx)}
            >
              {word}
            </Badge>
          ))}
        </div>
      </div>
    </div>
  );
}
