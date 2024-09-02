"use client";

import { Footer } from "@/src/components/footer";
import { HeroPredictor } from "@/src/components/hero-predictor";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";

export default function Home() {
  return (
    <main>
      <QueryClientProvider client={new QueryClient()}>
        <div className="px-2 py-10">
          <HeroPredictor />
        </div>
        <Footer />
      </QueryClientProvider>
    </main>
  );
}
