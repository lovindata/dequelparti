import nextConfig from "@/next.config.mjs";
import { components } from "@/services/vector_database/endpoints";
import { AutoModel, AutoTokenizer, env, cos_sim } from "@xenova/transformers";
import { useEffect, useState } from "react";

export function useAllMiniLML6V2() {
  const [{ allMiniLML6V2, isLoading, isPredicting }, setAllMiniLML6V2State] =
    useState<{
      allMiniLML6V2?: (
        text: string,
        embeddingTable: components["schemas"]["EmbeddingTableVo"],
      ) => {
        [key: components["schemas"]["LabelVo"]]: number;
      };
      isLoading: boolean;
      isPredicting: boolean;
    }>({
      allMiniLML6V2: undefined,
      isLoading: true,
      isPredicting: false,
    });

  useEffect(() => {
    env.remoteHost = `${window.location.origin}/${nextConfig.basePath}/artifacts/`;
    env.remotePathTemplate = "{model}";

    async function load() {
      const tokenizer = await AutoTokenizer.from_pretrained("all-MiniLM-L6-v2");
      const model = await AutoModel.from_pretrained("all-MiniLM-L6-v2", {
        quantized: false,
      });
      const allMiniLML6V2Loaded = (
        text: string,
        embeddingTable: components["schemas"]["EmbeddingTableVo"],
      ) => {};
      setAllMiniLML6V2State((curr) => ({
        ...curr,
        allMiniLML6V2: undefined,
        isLoading: false,
      }));
    }

    load();
  }, []);

  return { allMiniLML6V2, isLoading, isPredicting };
}
