/**
 * Загрузка туристической базы знаний из Wikivoyage и tool search_travel_info.
 * Аналог build_vectorstore + ti_retriever + search_travel_info из ноутбука (ячейки 1–17).
 */
import { convert } from "html-to-text";

/*
 * MemoryVectorStore — in-memory векторное хранилище из langchain/vectorstores/memory.
 * Используем его вместо Chroma чтобы не требовать внешний сервер для запуска секций ch11.
 * API совместим: similaritySearch / asRetriever.
 */
import { MemoryVectorStore } from "langchain/vectorstores/memory";

/*
 * RecursiveCharacterTextSplitter — рекурсивно режет текст по разделителям (\n\n, \n, " ").
 * Эквивалент RecursiveCharacterTextSplitter из langchain.text_splitter.
 */
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

/*
 * Document — пара (pageContent, metadata). Используется на всех этапах пайплайна.
 */
import { Document } from "@langchain/core/documents";

/*
 * tool — фабрика типобезопасных tool-ов для LangChain/LangGraph агентов.
 * Возвращает StructuredTool, который умеет описывать своё имя/схему/описание для bind_tools.
 */
import { tool, type StructuredToolInterface } from "@langchain/core/tools";
import { z } from "zod";

import { embeddings, UK_DESTINATIONS, WIKIVOYAGE_ROOT_URL } from "./config";

// Кэш-синглтон ретривера (ячейки #G–#K). Строим один раз, переиспользуем во всех секциях.
let _retriever: Awaited<ReturnType<MemoryVectorStore["asRetriever"]>> | null = null;

// Загружаем HTML, конвертируем в plain text, оборачиваем в Document (аналог AsyncHtmlLoader)
async function loadHtmlAsDoc(url: string): Promise<Document> {
  const response = await fetch(url);
  const html = await response.text();
  const text = convert(html, {
    wordwrap: false,
    selectors: [
      { selector: "a", options: { ignoreHref: true } },
      { selector: "img", format: "skip" },
      { selector: "script", format: "skip" },
      { selector: "style", format: "skip" },
    ],
  });
  return new Document({ pageContent: text, metadata: { source: url } });
}

// Аналог build_vectorstore() (ячейка #B): качаем страницы, режем на чанки, эмбеддим
async function buildVectorstore(destinations: string[]): Promise<MemoryVectorStore> {
  const urls = destinations.map((slug) => `${WIKIVOYAGE_ROOT_URL}/${slug}`);
  console.log("Скачиваем страницы направлений ...");
  const docs = await Promise.all(urls.map(loadHtmlAsDoc));

  // chunk_size=1024, chunk_overlap=128 — те же параметры, что в ноутбуке (ячейка #D)
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1024,
    chunkOverlap: 128,
  });
  const chunks: Document[] = [];
  for (const d of docs) {
    chunks.push(...(await splitter.splitDocuments([d])));
  }

  console.log(`Эмбеддим ${chunks.length} чанков ...`);
  const store = await MemoryVectorStore.fromDocuments(chunks, embeddings);
  console.log("Векторное хранилище готово.\n");
  return store;
}

// Singleton (ячейки #G–#I). Ленивая инициализация — впервые при вызове tool.
export async function getTravelInfoRetriever() {
  if (_retriever === null) {
    const store = await buildVectorstore(UK_DESTINATIONS);
    _retriever = store.asRetriever(4);
  }
  return _retriever;
}

// search_travel_info tool (ячейки 18–20).
// Получает query, делает similarity search, склеивает top-4 чанка через "\n---\n".
// StructuredToolInterface — широкий тип, чтобы TS не пытался выводить вглубь zod-схему.
export const searchTravelInfoTool: StructuredToolInterface = tool(
  async ({ query }: { query: string }) => {
    const retriever = await getTravelInfoRetriever();
    const docs = await retriever.invoke(query);
    const top = docs.slice(0, 4);
    return top.map((d) => d.pageContent).join("\n---\n");
  },
  {
    name: "search_travel_info",
    description: "Search travel information about destinations in England.",
    schema: z.object({
      query: z.string().describe("The query to search for in the travel info knowledge base."),
    }),
  }
);
