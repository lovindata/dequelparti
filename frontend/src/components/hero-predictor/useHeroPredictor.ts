import { useAllMiniLML6V2 } from "@/src/services/all-minilm-l6-v2";
import { useRef, useState } from "react";

export function useHeroPredictor() {
  const {
    predict,
    prediction,
    isLoadingVectorDatabase,
    isLoadingModel,
    isPredicting,
  } = useAllMiniLML6V2();
  const [userInput, setUserInput] = useState("");
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const setUserInputReducer = (userInput: string) => {
    timeoutRef.current !== null && clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => predict(userInput), 600);
    setUserInput(userInput);
  };

  return {
    userInput,
    setUserInput: setUserInputReducer,
    prediction,
    isLoadingVectorDatabase,
    isLoadingModel,
    isPredicting,
  };
}
