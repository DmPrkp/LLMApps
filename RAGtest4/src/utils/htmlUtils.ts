import * as cheerio from "cheerio";
import { convert } from "html-to-text";
import { ChromaClient } from "chromadb";

/*
 * Document — базовый класс для представления текстового документа в LangChain.
 * Содержит два поля: pageContent (текст) и metadata (произвольный объект с метаданными).
 * Используется на всех этапах пайплайна: загрузка, разбивка на чанки, хранение и retrieval.
 * Все векторные хранилища и retriever-ы принимают и возвращают массивы Document.
 */
import { Document } from "@langchain/core/documents";

// Загружаем HTML по URL и возвращаем как Document (аналог AsyncHtmlLoader)
export async function loadHtmlDocument(url: string): Promise<Document[]> {
  const response = await fetch(url);
  const html = await response.text();
  return [new Document({ pageContent: html, metadata: { source: url } })];
}

// Делим HTML на секции по заголовкам H1/H2, сохраняя только блоки под H2
// Аналог HTMLSectionSplitter с фильтром по h2 (ячейки 6–8 ноутбука)
export function splitDocsIntoGranularChunks(docs: Document[]): Document[] {
  const sections: Document[] = [];

  for (const doc of docs) {
    const $ = cheerio.load(doc.pageContent);
    let currentH1 = "";
    let currentH2 = "";
    let currentContent = "";

    const flush = () => {
      if (currentContent.trim() && currentH2) {
        sections.push(
          new Document({
            pageContent: currentContent.trim(),
            metadata: {
              ...doc.metadata,
              "Header 1": currentH1,
              "Header 2": currentH2,
            },
          })
        );
      }
      currentContent = "";
    };

    $("h1, h2, p, li").each((_, elem) => {
      const tag = (elem as { tagName?: string }).tagName?.toLowerCase();
      const text = $(elem).text().trim();
      if (!text) return;

      if (tag === "h1") {
        flush();
        currentH1 = text;
        currentH2 = "";
      } else if (tag === "h2") {
        flush();
        currentH2 = text;
        currentContent = text + "\n";
      } else {
        currentContent += text + "\n";
      }
    });

    flush();
  }

  return sections;
}

// Конвертируем HTML-документы в чистый текст (без тегов, скриптов, стилей)
export function htmlToTextDocs(docs: Document[]): Document[] {
  return docs.map((doc) => {
    const text = convert(doc.pageContent, {
      wordwrap: false,
      selectors: [
        { selector: "a", options: { ignoreHref: true } },
        { selector: "img", format: "skip" },
        { selector: "script", format: "skip" },
        { selector: "style", format: "skip" },
      ],
    });
    return new Document({ pageContent: text, metadata: doc.metadata });
  });
}

// Удаляем коллекцию Chroma перед повторной загрузкой данных
export async function resetChromaCollection(name: string): Promise<void> {
  const client = new ChromaClient();
  try {
    await client.deleteCollection({ name });
  } catch {
    // Коллекция ещё не существует — это нормально
  }
}
