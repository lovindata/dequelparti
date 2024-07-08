import * as math from "mathjs";
import vocabulary from "@/data/vocabulary.json";
import wordEmbeddingOfLemmas from "@/data/word_embedding_of_lemmas.json";

export function computeSlidedCosineSimilarity(
  input: string[],
  docLemmas: string[],
  window: number = 50,
  stride: number = 10
) {
  if (input.length === 0) return 0;

  const convertAsLemma = vocabulary as { [key: string]: string };
  const getLemmaVector = wordEmbeddingOfLemmas as { [key: string]: number[] };

  const inputLemmas = input.map((word) => convertAsLemma[word]);
  const inputVectors = inputLemmas.map((lemma) =>
    math.matrix(getLemmaVector[lemma])
  );
  const inputAvgVector = computeAvgVector(inputVectors);

  const docVectors = docLemmas.map((lemma) => {
    return math.matrix(getLemmaVector[lemma]);
  });
  const docSlidedVectors = buildSlidingWindow(docVectors, window, stride);
  const docSlidedAvgVector = docSlidedVectors.map(computeAvgVector);

  const cosineSimilarities = docSlidedAvgVector.map((avgVector) =>
    computeCosineSimilarity(avgVector, inputAvgVector)
  );
  const nbMatches =
    cosineSimilarities.filter((cosineSimilarity) => cosineSimilarity > 0.7)
      .length / cosineSimilarities.length;
  return nbMatches;
}

function computeAvgVector(vectors: math.Matrix[]) {
  const sumVector = vectors.reduce((vec0, vec1) =>
    math.add(math.matrix(vec0), math.matrix(vec1))
  );
  const avgVector = math.divide(sumVector, vectors.length) as math.Matrix;
  return avgVector;
}

function buildSlidingWindow<A>(arr: A[], window: number, stride: number) {
  const outArr: A[][] = [];
  for (let i = 0; i + window <= arr.length; i += stride) {
    // Extract the current window
    const slice = arr.slice(i, i + window);
    outArr.push(slice);
  }
  return outArr;
}

function computeCosineSimilarity(vector0: math.Matrix, vector1: math.Matrix) {
  const normVect1 = math.norm(vector0) as number;
  const normVect2 = math.norm(vector1) as number;
  return math.dot(vector0, vector1) / (normVect1 * normVect2);
}
