import { AutoModel, AutoTokenizer, env, cos_sim } from "@xenova/transformers";

async function run() {
  env.remoteHost =
    "https://raw.githubusercontent.com/lovindata/dequelparti/develop/frontend/public/artifacts/";
  env.remotePathTemplate = "{model}";
  const tokenizer = await AutoTokenizer.from_pretrained(
    "all-MiniLM-L6-v2",
    // "sentence-transformers/all-MiniLM-L6-v2",
  );
  const model = await AutoModel.from_pretrained(
    "all-MiniLM-L6-v2",
    // "sentence-transformers/all-MiniLM-L6-v2",
    { quantized: false },
  );

  async function getEmbedding(input) {
    let { input_ids, attention_mask } = await tokenizer(input);
    let output = await model({
      input_ids: input_ids,
      attention_mask: attention_mask,
    });
    return output.sentence_embedding.data;
  }

  const input = "That is a happy person";
  const docs = [
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day",
  ];

  const inputVector = await getEmbedding(input);
  console.log("############ INPUT ############");
  console.log(typeof inputVector, typeof inputVector[0]);
  console.log(inputVector);

  const docVectors = await Promise.all(docs.map((doc) => getEmbedding(doc)));
  console.log("############ DOCS ############");
  console.log(typeof docVectors, typeof docVectors[0], typeof docVectors[0][0]);
  console.log(docVectors);

  const scores = docVectors.map((docVector) => cos_sim(inputVector, docVector));
  console.log("############ SCORES ############");
  console.log(typeof scores, typeof scores[0]);
  console.log(scores);
}

run().catch(console.error);
