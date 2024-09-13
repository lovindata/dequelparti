"use client";

import { Footer } from "@/src/components/footer";
import { GitHubStats } from "@/src/components/github-stats";
import { HeroPredictor } from "@/src/components/hero-predictor";
import { ImageDivider } from "@/src/components/image-divider";
import { LogoWall } from "@/src/components/logo-wall";
import { Navbar } from "@/src/components/navbar";
import { ProgramsCarousel } from "@/src/components/programs-carousel";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col justify-between">
      <QueryClientProvider client={new QueryClient()}>
        <Navbar />
        <div className="mx-auto space-y-10 px-2 py-10 md:space-y-[72px] md:px-8 md:py-[72px]">
          <HeroPredictor />
        </div>
        <ImageDivider />
        <div className="space-y-10 px-2 py-10 md:space-y-[72px] md:px-8 md:py-[72px]">
          <ProgramsCarousel />
          <LogoWall />
          {/* <GitHubStats /> */}
        </div>
        <Footer />
      </QueryClientProvider>
    </main>
  );
}
