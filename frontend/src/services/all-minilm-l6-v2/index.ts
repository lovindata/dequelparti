import { useModel } from "@/src/services/all-minilm-l6-v2/useModel";
import { usePredictor } from "@/src/services/all-minilm-l6-v2/usePredictor";
import { useVectorDatabase } from "@/src/services/all-minilm-l6-v2/useVectorDatabase";
import { useCallback } from "react";

export function useAllMiniLML6V2() {
  const { vectorDatabase, isLoadingVectorDatabase } = useVectorDatabase();
  const { model, isLoadingModel } = useModel();
  const { predict, prediction, isPredicting } = usePredictor();

  const predictOrDoNothing = useCallback(
    (userInput: string) =>
      vectorDatabase !== undefined && model !== undefined
        ? predict(userInput, model, vectorDatabase)
        : undefined,
    [vectorDatabase, model, predict],
  );

  return {
    predict: predictOrDoNothing,
    prediction,
    isLoadingVectorDatabase,
    isLoadingModel,
    isPredicting,
  };
}
