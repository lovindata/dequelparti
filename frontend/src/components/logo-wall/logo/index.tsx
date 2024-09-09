import docker from "@/src/components/logo-wall/logo/assets/docker.webp";
import dockerDarkMode from "@/src/components/logo-wall/logo/assets/docker_dark_mode.webp";
import figma from "@/src/components/logo-wall/logo/assets/figma.webp";
import figmaDarkMode from "@/src/components/logo-wall/logo/assets/figma_dark_mode.webp";
import gemma2 from "@/src/components/logo-wall/logo/assets/gemma2.webp";
import gemma2DarkMode from "@/src/components/logo-wall/logo/assets/gemma2_dark_mode.webp";
import github from "@/src/components/logo-wall/logo/assets/github.webp";
import githubDarkMode from "@/src/components/logo-wall/logo/assets/github_dark_mode.webp";
import githubpages from "@/src/components/logo-wall/logo/assets/githubpages.webp";
import githubpagesDarkMode from "@/src/components/logo-wall/logo/assets/githubpages_dark_mode.webp";
import loguru from "@/src/components/logo-wall/logo/assets/loguru.webp";
import loguruDarkMode from "@/src/components/logo-wall/logo/assets/loguru_dark_mode.webp";
import nextjs from "@/src/components/logo-wall/logo/assets/nextjs.webp";
import nextjsDarkMode from "@/src/components/logo-wall/logo/assets/nextjs_dark_mode.webp";
import ollama from "@/src/components/logo-wall/logo/assets/ollama.webp";
import ollamaDarkMode from "@/src/components/logo-wall/logo/assets/ollama_dark_mode.webp";
import onnx from "@/src/components/logo-wall/logo/assets/onnx.webp";
import onnxDarkMode from "@/src/components/logo-wall/logo/assets/onnx_dark_mode.webp";
import pypdf2 from "@/src/components/logo-wall/logo/assets/pypdf2.webp";
import pypdf2DarkMode from "@/src/components/logo-wall/logo/assets/pypdf2_dark_mode.webp";
import pytorch from "@/src/components/logo-wall/logo/assets/pytorch.webp";
import pytorchDarkMode from "@/src/components/logo-wall/logo/assets/pytorch_dark_mode.webp";
import pytorchlightning from "@/src/components/logo-wall/logo/assets/pytorchlightning.webp";
import pytorchlightningDarkMode from "@/src/components/logo-wall/logo/assets/pytorchlightning_dark_mode.webp";
import react from "@/src/components/logo-wall/logo/assets/react.webp";
import reactDarkMode from "@/src/components/logo-wall/logo/assets/react_dark_mode.webp";
import scikitlearn from "@/src/components/logo-wall/logo/assets/scikitlearn.webp";
import scikitlearnDarkMode from "@/src/components/logo-wall/logo/assets/scikitlearn_dark_mode.webp";
import shadcnui from "@/src/components/logo-wall/logo/assets/shadcnui.webp";
import shadcnuiDarkMode from "@/src/components/logo-wall/logo/assets/shadcnui_dark_mode.webp";
import spacy from "@/src/components/logo-wall/logo/assets/spacy.webp";
import spacyDarkMode from "@/src/components/logo-wall/logo/assets/spacy_dark_mode.webp";
import tailwindcss from "@/src/components/logo-wall/logo/assets/tailwindcss.webp";
import tailwindcssDarkMode from "@/src/components/logo-wall/logo/assets/tailwindcss_dark_mode.webp";
import tqdm from "@/src/components/logo-wall/logo/assets/tqdm.webp";
import tqdmDarkMode from "@/src/components/logo-wall/logo/assets/tqdm_dark_mode.webp";
import transformers from "@/src/components/logo-wall/logo/assets/transformers.webp";
import transformersDarkMode from "@/src/components/logo-wall/logo/assets/transformers_dark_mode.webp";
import { useTheme } from "next-themes";
import Image, { StaticImageData } from "next/image";
import { useEffect, useMemo, useState } from "react";

interface Props {
  type:
    | "docker"
    | "figma"
    | "gemma2"
    | "github"
    | "githubpages"
    | "loguru"
    | "nextjs"
    | "ollama"
    | "onnx"
    | "pypdf2"
    | "pytorch"
    | "pytorchlightning"
    | "react"
    | "scikitlearn"
    | "shadcnui"
    | "spacy"
    | "tailwindcss"
    | "tqdm"
    | "transformers";
}

export function Logo({ type }: Props) {
  const { resolvedTheme } = useTheme();
  const logos: Record<
    Props["type"],
    { light: StaticImageData; dark: StaticImageData }
  > = useMemo(
    () => ({
      docker: { light: docker, dark: dockerDarkMode },
      figma: { light: figma, dark: figmaDarkMode },
      gemma2: { light: gemma2, dark: gemma2DarkMode },
      github: { light: github, dark: githubDarkMode },
      githubpages: { light: githubpages, dark: githubpagesDarkMode },
      loguru: { light: loguru, dark: loguruDarkMode },
      nextjs: { light: nextjs, dark: nextjsDarkMode },
      ollama: { light: ollama, dark: ollamaDarkMode },
      onnx: { light: onnx, dark: onnxDarkMode },
      pypdf2: { light: pypdf2, dark: pypdf2DarkMode },
      pytorch: { light: pytorch, dark: pytorchDarkMode },
      pytorchlightning: {
        light: pytorchlightning,
        dark: pytorchlightningDarkMode,
      },
      react: { light: react, dark: reactDarkMode },
      scikitlearn: { light: scikitlearn, dark: scikitlearnDarkMode },
      shadcnui: { light: shadcnui, dark: shadcnuiDarkMode },
      spacy: { light: spacy, dark: spacyDarkMode },
      tailwindcss: { light: tailwindcss, dark: tailwindcssDarkMode },
      tqdm: { light: tqdm, dark: tqdmDarkMode },
      transformers: { light: transformers, dark: transformersDarkMode },
    }),
    [],
  );
  const [src, setSrc] = useState<StaticImageData | undefined>(undefined);

  useEffect(() => {
    const logo = logos[type];
    const src = resolvedTheme === "dark" ? logo.dark : logo.light;
    setSrc(src);
  }, [logos, resolvedTheme, type]);

  return (
    src !== undefined && <Image src={src} alt={type} height={26} width={0} />
  );
}
