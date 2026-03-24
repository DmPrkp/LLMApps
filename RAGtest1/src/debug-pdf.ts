import * as dotenv from "dotenv";
dotenv.config();

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { textSplitter, PAESTUM_DIR } from "./utils/config";
import * as path from "path";

async function main() {
  const loader = new PDFLoader(path.join(PAESTUM_DIR, "PaestumRevisited.pdf"));
  const docs = await loader.load();
  const chunks = await textSplitter.splitDocuments(docs);

  console.log(`Total chunks: ${chunks.length}`);
  chunks.forEach((c, i) => {
    const content = c.pageContent.trim();
    const metaStr = JSON.stringify(c.metadata);
    const hasNull = metaStr.includes("null");
    const hasUndefined = metaStr.includes("undefined");
    if (content.length === 0 || hasNull || hasUndefined) {
      console.log(`[${i}] PROBLEM — empty:${content.length === 0} null:${hasNull} undef:${hasUndefined}`);
      console.log("  metadata:", metaStr);
      console.log("  content:", JSON.stringify(content.slice(0, 100)));
    }
  });
  console.log("Done scanning.");
}

main().catch(console.error);
