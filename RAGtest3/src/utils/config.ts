import * as dotenv from "dotenv";
import { OpenAIEmbeddings } from "@langchain/openai";

dotenv.config();

export const OPENAI_API_KEY = process.env.OPENAI_API_KEY ?? "";
export const GROQ_API_KEY = process.env.GROQ_API_KEY ?? "";

// Cell 1 — embeddings (original uses OpenAIEmbeddings)
export const embeddings = new OpenAIEmbeddings({
  openAIApiKey: OPENAI_API_KEY,
});

// Cell 4 — UK destinations list
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

// Cell 5 — destination URLs
export const ukDestinationUrls = UK_DESTINATIONS.map(
  (d) => `${WIKIVOYAGE_ROOT_URL}/${d}`
);
