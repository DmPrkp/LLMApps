import * as cheerio from "cheerio";
import { Document } from "@langchain/core/documents";
import { convert } from "html-to-text";
import { ChromaClient } from "chromadb";

/**
 * Fetches raw HTML from a URL and returns it as a Document array.
 * Equivalent to AsyncHtmlLoader (cells 6–7).
 */
export async function loadHtmlDocument(url: string): Promise<Document[]> {
  const response = await fetch(url);
  const html = await response.text();
  return [new Document({ pageContent: html, metadata: { source: url } })];
}

/**
 * Splits HTML documents into sections by H1/H2 headers,
 * keeping only H2-associated content.
 * Equivalent to HTMLSectionSplitter with h2 filter (cells 6–8).
 */
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

/**
 * Converts HTML documents to plain text.
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
 * Deletes a Chroma collection to recreate it from scratch.
 * Equivalent to collection.reset_collection() (cells 2–3).
 */
export async function resetChromaCollection(name: string): Promise<void> {
  const client = new ChromaClient();
  try {
    await client.deleteCollection({ name });
  } catch {
    // Collection did not exist yet — that's fine
  }
}
