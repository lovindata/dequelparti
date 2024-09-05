import nextConfig from "@/next.config.mjs";
import {
  AutoModel,
  AutoTokenizer,
  env,
  PreTrainedModel,
  PreTrainedTokenizer,
} from "@xenova/transformers";
import { useEffect, useState } from "react";

export function useModel() {
  const [model, setModel] = useState<
    { tokenizer: PreTrainedTokenizer; model: PreTrainedModel } | undefined
  >(undefined);

  useEffect(() => {
    async function load() {
      env.remoteHost = `${window.location.origin}${nextConfig.basePath}/artifacts/`;
      env.remotePathTemplate = "{model}";
      const tokenizer = await AutoTokenizer.from_pretrained("all-MiniLM-L6-v2");
      const model = await AutoModel.from_pretrained("all-MiniLM-L6-v2", {
        quantized: false,
      });
      setModel({ tokenizer, model });
    }
    load();
  }, []);

  return { model, isLoadingModel: model === undefined };
}
