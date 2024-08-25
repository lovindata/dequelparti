import { useInputWords } from "@/components/HeroPredictor/InputWords/useInputWords";
import axios from "axios";

export default function InputWords() {
  const { data, isLoading } = useInputWords();
  console.log(isLoading);
  console.log(data);
  return <div></div>;
}
