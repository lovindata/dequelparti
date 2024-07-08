"use client";

import ensembleTokens from "@/data/program_ensemble.json";
import frontPopulaireTokens from "@/data/program_nouveau_front_populaire.json";
import rassemblementNationalTokens from "@/data/program_rassemblement_national.json";
import { InputWords } from "@/components/input-words";
import { NavigationBar } from "@/components/navigation-bar";
import { ProgressBar } from "@/components/progress-bar";
import React from "react";
import { computeSlidedCosineSimilarity } from "@/helpers/nlp";

export default function Home() {
  const [words, setWords] = React.useState<string[]>([]);
  const appendWord = (word: string) => setWords((_) => [..._, word]);
  const popWordAt = (idx: number) =>
    setWords((_) => _.filter((_, currIdx) => currIdx != idx));
  const popAllWords = () => setWords((_) => []);

  let lcsEnsemble = computeSlidedCosineSimilarity(words, ensembleTokens);
  let lcsFrontPopulaire = computeSlidedCosineSimilarity(
    words,
    frontPopulaireTokens
  );
  let lcsRassemblementNational = computeSlidedCosineSimilarity(
    words,
    rassemblementNationalTokens
  );
  const lcsSum = lcsEnsemble + lcsFrontPopulaire + lcsRassemblementNational;
  lcsEnsemble = lcsSum && (lcsEnsemble / lcsSum) * 100;
  lcsFrontPopulaire = lcsSum && (lcsFrontPopulaire / lcsSum) * 100;
  lcsRassemblementNational =
    lcsSum && (lcsRassemblementNational / lcsSum) * 100;
  lcsEnsemble = Math.round(lcsEnsemble * 10) / 10;
  lcsFrontPopulaire = Math.round(lcsFrontPopulaire * 10) / 10;
  lcsRassemblementNational = Math.round(lcsRassemblementNational * 10) / 10;

  return (
    <main className="flex flex-col items-center justify-between p-8 sm:p-16 md:p-20 lg:p-24 space-y-8">
      <NavigationBar />
      <InputWords
        words={words}
        appendWord={appendWord}
        popWordAt={popWordAt}
        popAllWords={popAllWords}
      />
      <ProgressBar
        src="/dequelparti/imgs/pgrm_ensemble_page-0001.jpg"
        value={lcsEnsemble}
      />
      <ProgressBar
        src="/dequelparti/imgs/prgm_nouveau_front_populaire_page-0001.jpg"
        value={lcsFrontPopulaire}
      />
      <ProgressBar
        src="/dequelparti/imgs/prgm_rassemblement_national_page-0001.jpg"
        value={lcsRassemblementNational}
      />
    </main>
  );
}
