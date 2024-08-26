import nextConfig from "@/next.config.mjs";
import axios from "axios";

export function useVectorDatabase() {
  if (typeof window !== "undefined") {
    return axios.create({
      baseURL: `${window.location.origin}/${nextConfig.basePath}`,
    });
  }
  return axios.create();
}
