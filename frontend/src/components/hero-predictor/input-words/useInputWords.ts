import { useAllMiniLML6V2 } from "@/src/services/all-minilm-l6-v2";
import { useState } from "react";

export function useInputWords() {
  const {
    predict,
    prediction,
    isLoadingVectorDatabase,
    isLoadingModel,
    isPredicting,
  } = useAllMiniLML6V2();
  const [userInput, setUserInput] = useState("");

  const setUserInputReducer = (userInput: string) => {
    predict(userInput);
    setUserInput(userInput);
  };
  console.log("predict", predict);
  console.log("prediction", prediction);

  return {
    userInput,
    setUserInput: setUserInputReducer,
    prediction,
    isLoadingVectorDatabase,
    isLoadingModel,
    isPredicting,
  };
}
