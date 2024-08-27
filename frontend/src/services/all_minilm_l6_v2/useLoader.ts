import nextConfig from "@/next.config.mjs";
import {
  AutoModel,
  AutoTokenizer,
  env,
  PreTrainedModel,
  PreTrainedTokenizer,
} from "@xenova/transformers";
import { useEffect, useState } from "react";

export function useLoader() {
  const [loader, setLoaderState] = useState<
    { tokenizer: PreTrainedTokenizer; model: PreTrainedModel } | undefined
  >(undefined);

  useEffect(() => {
    env.remoteHost = `${window.location.origin}/${nextConfig.basePath}/artifacts/`;
    env.remotePathTemplate = "{model}";
    async function load() {
      const tokenizer = await AutoTokenizer.from_pretrained("all-MiniLM-L6-v2");
      const model = await AutoModel.from_pretrained("all-MiniLM-L6-v2", {
        quantized: false,
      });
      setLoaderState({ tokenizer, model });
    }
    load();
  }, []);

  return loader;
}
