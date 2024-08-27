import { useState } from "react";

export function usePredictor() {
  const [prediction, setPrediction] = useState<
    { [key: string]: number } | undefined
  >(undefined);

  const predict = (text: string) => {};
}
