import { useLazyLoader } from "@/services/lazyloader";
import { paths } from "@/services/lazyloader/endpoints";
import { useQuery } from "@tanstack/react-query";

export function useInputWords() {
  const lazyloader = useLazyLoader();
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
