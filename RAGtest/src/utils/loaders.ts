import * as fs from "fs";
import * as path from "path";

import { DocxLoader } from "@langchain/community/document_loaders/fs/docx";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { WikipediaQueryRun } from "@langchain/community/tools/wikipedia_query_run";
import { Document } from "@langchain/core/documents";
import { BaseDocumentLoader } from "langchain/document_loaders/base";

import { textSplitter, vectorDb, PAESTUM_DIR } from "./config";

export function getLoader(filePath: string): BaseDocumentLoader {
  const ext = path.extname(filePath).toLowerCase().slice(1);
  switch (ext) {
    case "pdf":  return new PDFLoader(filePath);
    case "docx": return new DocxLoader(filePath);
    case "txt":  return new TextLoader(filePath);
    default:     throw new Error(`No loader available for extension '${ext}'`);
  }
}

export async function splitAndImport(loader: BaseDocumentLoader, label: string): Promise<void> {
  const docs = await loader.load();
  const chunks = (await textSplitter.splitDocuments(docs))
    .filter(c => c.pageContent.trim().length > 0)
    .map(c => ({
      ...c,
      metadata: Object.fromEntries(
        Object.entries(c.metadata)
          .map(([k, v]) => [k, v === null || v === undefined ? "" : typeof v === "object" ? JSON.stringify(v) : v])
      ),
    }));
  await vectorDb.addDocuments(chunks);
  console.log(`Ingested ${chunks.length} chunks from: ${label}`);
}

export async function loadWikipedia(query: string): Promise<Document[]> {
  const tool = new WikipediaQueryRun({ topKResults: 1, maxDocContentLength: 10000 });
  const result = await tool.invoke(query);
  return [new Document({ pageContent: result, metadata: { source: `Wikipedia: ${query}` } })];
}

export async function ingestPaestum(): Promise<void> {
  console.log("\n=== Ingesting Paestum documents ===");

  const wikiDocs = await loadWikipedia("Paestum");
  const wikiChunks = (await textSplitter.splitDocuments(wikiDocs))
    .filter(c => c.pageContent.trim().length > 0);
  await vectorDb.addDocuments(wikiChunks);
  console.log(`Ingested ${wikiChunks.length} chunks from: Wikipedia – Paestum`);

  await splitAndImport(new DocxLoader(path.join(PAESTUM_DIR, "Paestum-Britannica.docx")), "Paestum-Britannica.docx");
  await splitAndImport(new PDFLoader(path.join(PAESTUM_DIR, "PaestumRevisited.pdf")), "PaestumRevisited.pdf");
  await splitAndImport(new TextLoader(path.join(PAESTUM_DIR, "Paestum-Encyclopedia.txt")), "Paestum-Encyclopedia.txt");
}

export async function ingestFolder(folderPath: string): Promise<void> {
  console.log(`\n=== Ingesting folder: ${folderPath} ===`);
  const files = fs.readdirSync(folderPath);
  for (const filename of files) {
    const filePath = path.join(folderPath, filename);
    if (!fs.statSync(filePath).isFile()) continue;
    try {
      await splitAndImport(getLoader(filePath), filename);
    } catch (err: unknown) {
      if (err instanceof Error) console.log(`Skipping ${filename}: ${err.message}`);
    }
  }
}
