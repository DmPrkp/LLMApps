import * as dotenv from "dotenv";

/*
 * HuggingFaceTransformersEmbeddings — локальная модель для создания векторных эмбеддингов.
 * Работает полностью на CPU без внешних API через ONNX Runtime.
 * Используется модель Xenova/all-MiniLM-L6-v2: лёгкая, быстрая, 384-мерные векторы.
 * Эмбеддинги нужны для преобразования текста в векторы перед сохранением в Chroma.
 */
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";

dotenv.config();

export const GROQ_API_KEY = process.env.GROQ_API_KEY ?? "";

export const embeddings = new HuggingFaceTransformersEmbeddings({
  model: "Xenova/bge-base-en-v1.5",
});

export const WIKIVOYAGE_ROOT_URL = "https://en.wikivoyage.org/wiki";

// Маппинг направление → регион (ячейки 6-8 из ноутбука ch10)
export const UK_DESTINATION_REGIONS: Record<string, string> = {
  Cornwall: "Cornwall",
  North_Cornwall: "Cornwall",
  South_Cornwall: "Cornwall",
  West_Cornwall: "Cornwall",
  Tintagel: "Cornwall",
  Bodmin: "Cornwall",
  Wadebridge: "Cornwall",
  Penzance: "Cornwall",
  Newquay: "Cornwall",
  St_Ives: "Cornwall",
  Port_Isaac: "Cornwall",
  Looe: "Cornwall",
  Polperro: "Cornwall",
  Porthleven: "Cornwall",
  East_Sussex: "East Sussex",
  Brighton: "East Sussex",
  Battle: "East Sussex",
  "Hastings_(England)": "East Sussex",
  "Rye_(England)": "East Sussex",
  Seaford: "East Sussex",
  Ashdown_Forest: "East Sussex",
};

export const ukDestinationUrls = Object.keys(UK_DESTINATION_REGIONS).map(
  (d) => `${WIKIVOYAGE_ROOT_URL}/${d}`
);
