import * as path from "path";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

export const DATA_ROOT = path.resolve(__dirname, "../../");
export const PAESTUM_DIR = path.join(DATA_ROOT, "Paestum");
export const CILENTO_DIR = path.join(DATA_ROOT, "CilentoTouristInfo");

export const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 0,
});

export const embeddingsModel = new HuggingFaceTransformersEmbeddings({
  model: "Xenova/all-MiniLM-L6-v2",
});

export const vectorDb = new Chroma(embeddingsModel, {
  collectionName: "tourist_info_hf4",
});
