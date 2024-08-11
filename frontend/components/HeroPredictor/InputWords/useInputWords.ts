import { useQuery } from "@tanstack/react-query";
import axios from "axios";

export function useInputWords() {
  axios.get("/dequelparti/artifacts/dataeng/vocabulary/a.json");
}
