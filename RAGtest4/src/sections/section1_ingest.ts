/**
 * Секция 1: Загрузка данных с метаданными
 * Ячейки 1–17: Загружаем HTML со страниц Wikivoyage, добавляем метаданные
 * destination+region к каждому чанку и сохраняем в векторное хранилище Chroma.
 */

/*
 * Chroma — интеграция с векторным хранилищем ChromaDB.
 * Позволяет сохранять документы как векторные эмбеддинги и искать по семантическому сходству.
 * Поддерживает фильтрацию по метаданным через операторы $eq, $and, $or.
 * Используется как основное хранилище для RAG-поиска во всех секциях.
 */
import { Chroma } from "@langchain/community/vectorstores/chroma";

/*
 * Document — базовый класс для представления текстового документа в LangChain.
 * Содержит два поля: pageContent (текст) и metadata (произвольный объект с метаданными).
 * Используется на всех этапах пайплайна: загрузка, разбивка на чанки, хранение и retrieval.
 * Все векторные хранилища и retriever-ы принимают и возвращают массивы Document.
 */
import { Document } from "@langchain/core/documents";

import {
  embeddings,
  UK_DESTINATION_REGIONS,
  WIKIVOYAGE_ROOT_URL,
} from "../utils/config";
import {
  loadHtmlDocument,
  splitDocsIntoGranularChunks,
  resetChromaCollection,
} from "../utils/htmlUtils";

export const COLLECTION_NAME = "uk_metadata";

export async function runDataIngestion() {
  console.log("\n========== ① Загрузка данных с метаданными ==========");

  // Ячейка 2 — сбрасываем коллекцию перед повторной загрузкой
  await resetChromaCollection(COLLECTION_NAME);

  const collection = new Chroma(embeddings, {
    collectionName: COLLECTION_NAME,
  });

  // Ячейки 9–17 — загружаем каждый URL, добавляем метаданные, делим на чанки
  for (const [destination, region] of Object.entries(UK_DESTINATION_REGIONS)) {
    const url = `${WIKIVOYAGE_ROOT_URL}/${destination}`;
    const docs = await loadHtmlDocument(url);
    console.log(`Загружаем: ${url}`);

    const chunks = splitDocsIntoGranularChunks(docs);

    // Ячейки 12–14 — обогащаем каждый чанк полями destination и region
    const chunksWithMeta = chunks.map(
      (doc) =>
        new Document({
          pageContent: doc.pageContent,
          metadata: {
            ...doc.metadata,
            destination: destination.replace(/_/g, " ").replace(/\(.*?\)/g, "").trim(),
            region,
          },
        })
    );

    await collection.addDocuments(chunksWithMeta);
  }

  console.log("\nЗагрузка завершена.");
}

async function main() {
  await runDataIngestion();
}

if (require.main === module) {
  main().catch(console.error);
}
