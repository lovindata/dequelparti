import {
  cos_sim,
  PreTrainedModel,
  PreTrainedTokenizer,
  softmax,
} from "@xenova/transformers";
import { useCallback, useState } from "react";

export function usePredictor() {
  const [prediction, setPrediction] = useState<
    { [key: string]: number } | undefined
  >(undefined);
  const [isPredicting, setIsPredicting] = useState(false);
  const predict = useCallback(
    (
      userInput: string,
      model: { tokenizer: PreTrainedTokenizer; model: PreTrainedModel },
      vectorDatabase: {
        id: number;
        decoded_embedding: string;
        vector_embedding: number[];
        label: string;
      }[],
    ) => {
      const uniqueLabels = Array.from(
        new Set(vectorDatabase.map((embeddingRow) => embeddingRow.label)),
      ).sort();
      if (userInput === "") {
        setPrediction(
          uniqueLabels.reduce(
            (output, label) => {
              output[label] = 0;
              return output;
            },
            {} as { [key: string]: number },
          ),
        );
        setIsPredicting(false);
        return;
      }

      async function predictAsync() {
        const { input_ids, attention_mask } = await model.tokenizer(userInput);
        const userInputVectorEmbedding: number[] = await model
          .model({
            input_ids: input_ids,
            attention_mask: attention_mask,
          })
          .then(
            (_: { sentence_embedding: { data: number[] } }) =>
              _.sentence_embedding.data,
          );
        const scores = vectorDatabase.map((embeddingRow) => ({
          label: embeddingRow.label,
          cosSimScore: cos_sim(
            userInputVectorEmbedding,
            embeddingRow.vector_embedding,
          ),
        }));
        const maxCosSimScores = uniqueLabels.map((uniqueLabel) =>
          Math.max(
            ...scores
              .filter(({ label }) => label === uniqueLabel)
              .map(({ cosSimScore }) => cosSimScore),
          ),
        );
        const softmaxScores = softmax(maxCosSimScores.map((x) => x / 0.05));
        const maxScores = uniqueLabels.reduce(
          (output, label, i) => {
            output[label] = softmaxScores[i];
            return output;
          },
          {} as { [key: string]: number },
        );
        setPrediction(maxScores);
        setIsPredicting(false);
      }
      setIsPredicting(true);
      predictAsync();
    },
    [],
  );

  return { predict, prediction, isPredicting };
}
