import * as dotenv from "dotenv";

/*
 * HuggingFaceTransformersEmbeddings — локальная модель для эмбеддингов на CPU.
 * Используем Xenova/bge-base-en-v1.5 — 768-мерные векторы, без внешних API.
 */
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";

/*
 * ChatOpenAI — обёртка над OpenAI Chat API и совместимыми эндпоинтами (Groq).
 * Поддерживает bind_tools, with_structured_output, function calling.
 */
import { ChatOpenAI } from "@langchain/openai";

dotenv.config();

export const GROQ_API_KEY = process.env.GROQ_API_KEY ?? "";

if (!GROQ_API_KEY) {
  console.warn("[config] GROQ_API_KEY не задан. Запусти `cp .env_example .env` и подставь ключ.");
}

// Список UK-направлений для туристической базы знаний (ячейки #A)
export const UK_DESTINATIONS = [
  "Cornwall",
  "North_Cornwall",
  "South_Cornwall",
  "West_Cornwall",
];

export const WIKIVOYAGE_ROOT_URL = "https://en.wikivoyage.org/wiki";

// Локальные эмбеддинги (заменяют OpenAIEmbeddings из ноутбука)
export const embeddings = new HuggingFaceTransformersEmbeddings({
  model: "Xenova/bge-base-en-v1.5",
});

// Фабрика LLM (Groq через OpenAI-совместимый endpoint)
// Все секции вызывают getLlm() чтобы делить один и тот же конфиг.
export function getLlm(modelName: string = "llama-3.3-70b-versatile"): ChatOpenAI {
  return new ChatOpenAI({
    modelName,
    openAIApiKey: GROQ_API_KEY,
    temperature: 0,
    configuration: { baseURL: "https://api.groq.com/openai/v1" },
  });
}
