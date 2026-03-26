import * as dotenv from "dotenv";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

dotenv.config();

export const OPENAI_API_KEY = process.env.OPENAI_API_KEY ?? "";

export const embeddings = new OpenAIEmbeddings({
  openAIApiKey: OPENAI_API_KEY,
});

// ячейка 24
export const UK_DESTINATIONS = [
  "Cornwall",
  "North_Cornwall",
  "South_Cornwall",
  "West_Cornwall",
  "Tintagel",
  "Bodmin",
  "Wadebridge",
  "Penzance",
  "Newquay",
  "St_Ives",
  "Port_Isaac",
  "Looe",
  "Polperro",
  "Porthleven",
  "East_Sussex",
  "Brighton",
  "Battle",
  "Hastings_(England)",
  "Rye_(England)",
  "Seaford",
  "Ashdown_Forest",
];

export const WIKIVOYAGE_ROOT_URL = "https://en.wikivoyage.org/wiki";

// ячейка 25
export const ukDestinationUrls = UK_DESTINATIONS.map(
  (d) => `${WIKIVOYAGE_ROOT_URL}/${d}`
);

// ячейка 17
export const coarseTextSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 3000,
  chunkOverlap: 300,
});

// ячейка 32 — родительский/дочерний сплиттеры
export const parentSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 3000,
});

export const childSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
});

// ячейка 79 — гранулярный сплиттер для стратегии расширенного контекста
export const granularChunkSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
});
