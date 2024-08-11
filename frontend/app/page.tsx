"use client";

import HeroPredictor from "@/components/HeroPredictor";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";

export default function Home() {
  return (
    <main className="px-2">
      <QueryClientProvider client={new QueryClient()}>
        <HeroPredictor />
      </QueryClientProvider>
    </main>
  );
}
