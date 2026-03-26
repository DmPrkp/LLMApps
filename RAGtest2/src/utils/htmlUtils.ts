import * as cheerio from "cheerio";
import { Document } from "@langchain/core/documents";
import { convert } from "html-to-text";
import { ChromaClient } from "chromadb";

/**
 * Загружает сырой HTML и возвращает Document.
 * Эквивалент AsyncHtmlLoader (ячейки 6–7).
 */
export async function loadHtmlDocument(url: string): Promise<Document[]> {
  const response = await fetch(url);
  const html = await response.text();
  return [new Document({ pageContent: html, metadata: { source: url } })];
}

/**
 * Конвертирует HTML-документы в простой текст.
 * Эквивалент Html2TextTransformer (ячейки 16, 18).
 */
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

/**
 * Разбивает HTML-документы на чанки по границам заголовков H1/H2.
 * Эквивалент HTMLSectionSplitter (ячейки 10–11).
 */
export function splitByHtmlSections(docs: Document[]): Document[] {
  const sections: Document[] = [];

  for (const doc of docs) {
    const $ = cheerio.load(doc.pageContent);
    let currentHeader = "";
    let currentContent = "";

    const flush = () => {
      if (currentContent.trim()) {
        sections.push(
          new Document({
            pageContent: currentContent.trim(),
            metadata: { ...doc.metadata, "Header 1": currentHeader },
          })
        );
      }
      currentContent = "";
    };

    $("h1, h2, p, li").each((_, elem) => {
      const tag = (elem as { tagName?: string }).tagName?.toLowerCase();
      const text = $(elem).text().trim();
      if (!text) return;

      if (tag === "h1" || tag === "h2") {
        flush();
        currentHeader = text;
        currentContent = text + "\n";
      } else {
        currentContent += text + "\n";
      }
    });

    flush();
  }

  return sections;
}

/**
 * Удаляет коллекцию Chroma, чтобы пересоздать её с нуля.
 * Эквивалент collection.reset_collection() (ячейки 2–3, 22–23 и др.).
 */
export async function resetChromaCollection(name: string): Promise<void> {
  const client = new ChromaClient();
  try {
    await client.deleteCollection({ name });
  } catch {
    // Коллекция ещё не существовала — это нормально
  }
}
