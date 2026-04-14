/**
 * Section 1: Data Ingestion
 * Cells 1–8: Load HTML from Wikivoyage URLs and split into granular H2 chunks,
 * then add to Chroma vector store.
 */
import { Chroma } from "@langchain/community/vectorstores/chroma";

import { embeddings, ukDestinationUrls } from "../utils/config";
import {
  loadHtmlDocument,
  splitDocsIntoGranularChunks,
  resetChromaCollection,
} from "../utils/htmlUtils";

async function main() {
  console.log("=== Section 1: Data Ingestion ===\n");

  // Cell 2 — reset collection
  await resetChromaCollection("uk_granular");

  const ukGranularCollection = new Chroma(embeddings, {
    collectionName: "uk_granular",
  });

  // Cell 8 — load each destination and add granular H2 chunks
  for (const destinationUrl of ukDestinationUrls) {
    const docs = await loadHtmlDocument(destinationUrl); // cells 6–7 #E #F
    console.log(`Loaded: ${destinationUrl}`);

    const granularChunks = splitDocsIntoGranularChunks(docs); // #B #C #D
    await ukGranularCollection.addDocuments(granularChunks);
  }

  console.log("\nIngestion complete.");
}

main().catch(console.error);
