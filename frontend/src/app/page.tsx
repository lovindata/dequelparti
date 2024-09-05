"use client";

import { Footer } from "@/src/components/footer";
import { HeroPredictor } from "@/src/components/hero-predictor";
import { Navbar } from "@/src/components/navbar";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col justify-between">
      <QueryClientProvider client={new QueryClient()}>
        <Navbar />
        <div className="px-2 py-10 md:px-8 md:py-[72px]">
          <HeroPredictor />
        </div>
        <Footer />
      </QueryClientProvider>
    </main>
  );
}
