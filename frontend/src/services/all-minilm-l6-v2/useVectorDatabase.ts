import nextConfig from "@/next.config.mjs";
import { useQuery } from "@tanstack/react-query";
import axios from "axios";

export function useVectorDatabase() {
  const { data: vectorDatabase, isLoading: isLoadingVectorDatabase } = useQuery(
    {
      queryKey: [`${nextConfig.basePath}/artifacts/vector-database.json`],
      queryFn: () =>
        axios
          .get<
            {
              id: number;
              decoded_embedding: string;
              vector_embedding: number[];
              label: string;
            }[]
          >(`${nextConfig.basePath}/artifacts/vector-database.json`)
          .then((response) => response.data),
    },
  );
  return { vectorDatabase, isLoadingVectorDatabase };
}
