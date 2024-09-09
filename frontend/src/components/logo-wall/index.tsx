import { Logo } from "@/src/components/logo-wall/logo";
import { Section } from "@/src/components/shared/atoms/section";

export function LogoWall() {
  return (
    <Section
      title="Une IA avant-gardiste"
      subTitle="Les toutes dernières technologies de pointe compilées dans ce projet unique, entre vos mains !"
      className="flex flex-wrap justify-center gap-5"
    >
      <Logo type="react" />
      <Logo type="tailwindcss" />
      <Logo type="shadcnui" />
      <Logo type="nextjs" />
      <Logo type="figma" />
      <Logo type="pytorch" />
      <Logo type="pytorchlightning" />
      <Logo type="transformers" />
      <Logo type="scikitlearn" />
      <Logo type="spacy" />
      <Logo type="pypdf2" />
      <Logo type="tqdm" />
      <Logo type="loguru" />
      <Logo type="onnx" />
      <Logo type="ollama" />
      <Logo type="gemma2" />
      <Logo type="docker" />
      <Logo type="githubpages" />
    </Section>
  );
}
