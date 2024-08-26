import { useVectorDatabase } from "@/services/vector_database";
import { paths } from "@/services/vector_database/endpoints";
import { useQuery } from "@tanstack/react-query";

export function useInputWords() {
  const lazyloader = useVectorDatabase();
  const { data, isLoading } = useQuery({
    queryKey: ["/artifacts/vector-database.json"],
    queryFn: () =>
      lazyloader
        .get<
          paths["/artifacts/vector-database.json"]["get"]["responses"]["200"]["content"]["application/json"]
        >("/artifacts/vector-database.json")
        .then((response) => response.data),
  });
  return { data, isLoading };
}
