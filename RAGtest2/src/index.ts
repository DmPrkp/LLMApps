import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Document } from "@langchain/core/documents";
import { ParentDocumentRetriever } from "langchain/retrievers/parent_document";
import { MultiVectorRetriever } from "langchain/retrievers/multi_vector";
import { InMemoryStore } from "langchain/storage/in_memory";
import { v4 as uuidv4 } from "uuid";
import { z } from "zod";

import {
  embeddings,
  OPENAI_API_KEY,
  coarseTextSplitter,
  parentSplitter,
  childSplitter,
  granularChunkSplitter,
  ukDestinationUrls,
} from "./utils/config";
import {
  loadHtmlDocument,
  htmlToTextDocs,
  splitByHtmlSections,
  resetChromaCollection,
} from "./utils/htmlUtils";

// ─────────────────────────────────────────────────────────────────────────────
// ①  Гранулярное vs Крупное разбиение — один URL Cornwall (ячейки 1–21)
// ─────────────────────────────────────────────────────────────────────────────
async function runGranularVsCoarseSingleUrl() {
  console.log("\n========== ① Granular vs Coarse (Cornwall only) ==========");

  // ячейки 2–3: создаём / сбрасываем коллекции
  await resetChromaCollection("cornwall_granular");
  await resetChromaCollection("cornwall_coarse");

  const cornwallGranularCollection = new Chroma(embeddings, { // #A
    collectionName: "cornwall_granular",
  });
  const cornwallCoarseCollection = new Chroma(embeddings, { // #A
    collectionName: "cornwall_coarse",
  });

  // ячейки 5–7: загружаем HTML
  const destinationUrl = "https://en.wikivoyage.org/wiki/Cornwall"; // ячейка 5
  const docs = await loadHtmlDocument(destinationUrl); // ячейки 6–7
  console.log(`Loaded ${docs.length} doc(s) from ${destinationUrl}`);

  // ячейки 11–13: гранулярное разбиение и загрузка
  const granularChunks = splitByHtmlSections(docs); // #B
  console.log(`Granular chunks: ${granularChunks.length}`);
  await cornwallGranularCollection.addDocuments(granularChunks);

  // ячейки 18–20: крупное разбиение и загрузка
  const textDocs = htmlToTextDocs(docs); // #A
  const coarseChunks = await coarseTextSplitter.splitDocuments(textDocs);
  console.log(`Coarse chunks: ${coarseChunks.length}`);
  await cornwallCoarseCollection.addDocuments(coarseChunks);

  // ячейка 14: гранулярный поиск
  console.log('\n--- Granular search: "Events or festivals in Cornwall" ---');
  const granularResults = await cornwallGranularCollection.similaritySearch(
    "События или фестивали в Корнуолле",
    3
  );
  for (const doc of granularResults) console.log(doc);

  // ячейка 21: крупный поиск
  console.log('\n--- Coarse search: "Events or festivals in Cornwall" ---');
  const coarseResults = await cornwallCoarseCollection.similaritySearch(
    "События или фестивали в Корнуолле",
    3
  );
  for (const doc of coarseResults) console.log(doc);
}

// ─────────────────────────────────────────────────────────────────────────────
// ②  Загрузка множества URL — гранулярное vs крупное разбиение (ячейки 22–30)
// ─────────────────────────────────────────────────────────────────────────────
async function runMultiUrlGranularVsCoarse() {
  console.log("\n========== ② Multi-URL Granular vs Coarse ==========");

  // ячейки 22–23: создаём / сбрасываем
  await resetChromaCollection("uk_granular");
  await resetChromaCollection("uk_coarse");

  const ukGranularCollection = new Chroma(embeddings, { // #A
    collectionName: "uk_granular",
  });
  const ukCoarseCollection = new Chroma(embeddings, { // #A
    collectionName: "uk_coarse",
  });

  // ячейка 26: пакетная загрузка
  for (const destinationUrl of ukDestinationUrls) {
    const docs = await loadHtmlDocument(destinationUrl); // #C #D
    console.log(`Ingesting ${destinationUrl} (${docs[0]?.metadata?.source})`);

    const granularChunks = splitByHtmlSections(docs);
    await ukGranularCollection.addDocuments(granularChunks);

    const textDocs = htmlToTextDocs(docs);
    const coarseChunks = await coarseTextSplitter.splitDocuments(textDocs);
    await ukCoarseCollection.addDocuments(coarseChunks);
  }

  // ячейка 27: гранулярный поиск
  console.log('\n--- Granular search: "Events or festivals in East Sussex" ---');
  const granularResults = await ukGranularCollection.similaritySearch(
    "События или фестивали в Восточном Суссексе",
    4
  );
  for (const doc of granularResults) console.log(doc);

  // ячейка 28: крупный поиск
  console.log('\n--- Coarse search: "Events or festivals in East Sussex" ---');
  const coarseResults = await ukCoarseCollection.similaritySearch(
    "События или фестивали в Восточном Суссексе",
    4
  );
  for (const doc of coarseResults) console.log(doc);

  // ячейки 29–30: запрос о пляжах
  console.log('\n--- Granular search: "Beaches in Cornwall" ---');
  const granularBeaches = await ukGranularCollection.similaritySearch(
    "Пляжи в Корнуолле",
    4
  );
  for (const doc of granularBeaches) console.log(doc);

  console.log('\n--- Coarse search: "Beaches in Cornwall" ---');
  const coarseBeaches = await ukCoarseCollection.similaritySearch(
    "Пляжи в Корнуолле",
    4
  );
  for (const doc of coarseBeaches) console.log(doc);
}

// ─────────────────────────────────────────────────────────────────────────────
// ③  Ретривер родительских документов (ячейки 31–41)
// ─────────────────────────────────────────────────────────────────────────────
async function runParentDocumentRetriever() {
  console.log("\n========== ③ Parent Document Retriever ==========");

  await resetChromaCollection("uk_child_chunks"); // #D

  const childChunksCollection = new Chroma(embeddings, { // #C
    collectionName: "uk_child_chunks",
  });

  const docStore = new InMemoryStore<Document>(); // #E

  const parentDocRetriever = new ParentDocumentRetriever({ // #F
    vectorstore: childChunksCollection,
    docstore: docStore,
    childSplitter,
    parentSplitter,
  });

  // ячейка 33: загрузка
  for (const destinationUrl of ukDestinationUrls) {
    const htmlDocs = await loadHtmlDocument(destinationUrl); // #A #B
    const textDocs = htmlToTextDocs(htmlDocs); // #C
    console.log(`Ingesting ${destinationUrl}`);
    await parentDocRetriever.addDocuments(textDocs, { ids: undefined }); // #D
  }

  // ячейка 34: список ключей
  const keys: string[] = [];
  for await (const key of docStore.yieldKeys()) keys.push(key);
  console.log(`\nStored keys: ${keys.length}`);

  // ячейка 35: извлечение
  const retrievedDocs = await parentDocRetriever.invoke("Корнуолл Рейнджер");
  console.log(`\nRetrieved ${retrievedDocs.length} parent doc(s)`); // ячейка 36

  // ячейка 37: первый документ
  if (retrievedDocs.length > 0) console.log(retrievedDocs[0]);

  // ячейки 38–40: сравнение прямого поиска по дочерним чанкам
  const childDocsOnly = await childChunksCollection.similaritySearch("Корнуолл Рейнджер");
  console.log(`\nDirect child search: ${childDocsOnly.length} chunk(s)`);
  if (childDocsOnly.length > 0) console.log(childDocsOnly[0]);

  // ячейка 41: ВАЖНО — гранулярный поиск находит чанк, но теряет окружающий контекст
}

// ─────────────────────────────────────────────────────────────────────────────
// ④  Мульти-векторный ретривер с дочерними чанками (ячейки 42–52)
// ─────────────────────────────────────────────────────────────────────────────
async function runMultiVectorChildChunks() {
  console.log("\n========== ④ Multi-Vector Retriever — Child Chunks ==========");

  await resetChromaCollection("uk_child_chunks_mv"); // #D

  const childChunksCollection = new Chroma(embeddings, { // #C
    collectionName: "uk_child_chunks_mv",
  });

  const docByteStore = new InMemoryStore<Uint8Array>(); // #E
  const docKey = "doc_id";

  const multiVectorRetriever = new MultiVectorRetriever({ // #F
    vectorstore: childChunksCollection,
    byteStore: docByteStore,
    idKey: docKey,
  });

  // ячейка 44: загрузка
  for (const destinationUrl of ukDestinationUrls) {
    const htmlDocs = await loadHtmlDocument(destinationUrl); // #A #B
    const textDocs = htmlToTextDocs(htmlDocs); // #C

    const coarseChunks = await parentSplitter.splitDocuments(textDocs); // #D
    const coarseChunkIds = coarseChunks.map(() => uuidv4());

    const allGranularChunks: Document[] = [];

    for (let i = 0; i < coarseChunks.length; i++) { // #E
      const coarseChunkId = coarseChunkIds[i];
      const granularChunks = await childSplitter.splitDocuments([coarseChunks[i]]); // #F

      for (const granularChunk of granularChunks) {
        granularChunk.metadata[docKey] = coarseChunkId; // #G
      }
      allGranularChunks.push(...granularChunks);
    }

    console.log(`Ingesting ${destinationUrl}`);
    await multiVectorRetriever.vectorstore.addDocuments(allGranularChunks); // #H
    await multiVectorRetriever.docstore.mset(
      coarseChunkIds.map((id, i) => [id, coarseChunks[i]]) // #I
    );
  }

  // ячейка 45: извлечение
  const retrievedDocs = await multiVectorRetriever.invoke("Корнуолл Рейнджер");
  console.log(`\nRetrieved ${retrievedDocs.length} parent doc(s)`); // ячейка 46
  // ВАЖНО: аналогично ParentDocumentRetriever, но с большим контролем и гибкостью.
  if (retrievedDocs.length > 0) console.log(retrievedDocs[0]); // ячейка 47

  // ячейки 49–51: прямой поиск по дочерним чанкам
  const childDocsOnly = await childChunksCollection.similaritySearch("Корнуолл Рейнджер");
  console.log(`\nDirect child search: ${childDocsOnly.length} chunk(s)`);
  if (childDocsOnly.length > 0) console.log(childDocsOnly[0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// ⑤  Мульти-векторный ретривер с LLM-резюме (ячейки 53–65)
// ─────────────────────────────────────────────────────────────────────────────
async function runMultiVectorSummaries() {
  console.log("\n========== ⑤ Multi-Vector Retriever — Summaries ==========");

  await resetChromaCollection("uk_summaries"); // #C

  const summariesCollection = new Chroma(embeddings, { // #B
    collectionName: "uk_summaries",
  });

  const docByteStore = new InMemoryStore<Uint8Array>(); // #D
  const docKey = "doc_id";

  const multiVectorRetriever = new MultiVectorRetriever({ // #E
    vectorstore: summariesCollection,
    byteStore: docByteStore,
    idKey: docKey,
  });

  // ячейка 55: LLM
  const llm = new ChatOpenAI({ // ячейка 55
    model: "gpt-4o-mini",
    openAIApiKey: OPENAI_API_KEY,
  });

  // ячейка 56: цепочка суммаризации
  const summarizationChain = ChatPromptTemplate.fromTemplate( // #B
    "Кратко изложи следующий документ:\n\n{document}"
  )
    .pipe(llm) // #A лямбда становится маппингом входных данных в промпт
    .pipe(new StringOutputParser());

  // ячейка 57: загрузка с резюме
  for (const destinationUrl of ukDestinationUrls) {
    const htmlDocs = await loadHtmlDocument(destinationUrl); // #A #B
    const textDocs = htmlToTextDocs(htmlDocs); // #C

    const coarseChunks = await parentSplitter.splitDocuments(textDocs); // #D
    const coarseChunkIds = coarseChunks.map(() => uuidv4());

    const allSummaries: Document[] = [];

    for (let i = 0; i < coarseChunks.length; i++) { // #E
      const coarseChunkId = coarseChunkIds[i];

      const summaryText = await summarizationChain.invoke({ // #F
        document: coarseChunks[i].pageContent,
      });

      const summaryDoc = new Document({
        pageContent: summaryText as string,
        metadata: { [docKey]: coarseChunkId },
      });
      allSummaries.push(summaryDoc); // #G
    }

    // ПРИМЕЧАНИЕ: медленнее стратегии с дочерними чанками из-за вызова LLM для каждого чанка.
    // Внешний цикл можно распараллелить для ускорения.
    console.log(`Ingesting ${destinationUrl}`);
    await multiVectorRetriever.vectorstore.addDocuments(allSummaries); // #H
    await multiVectorRetriever.docstore.mset(
      coarseChunkIds.map((id, i) => [id, coarseChunks[i]]) // #I
    );
  }

  // ячейка 59: извлечение через резюме
  const retrievedDocs = await multiVectorRetriever.invoke("Путешествие по Корнуоллу");
  console.log(`\nRetrieved ${retrievedDocs.length} parent doc(s)`); // ячейка 60
  console.log(retrievedDocs); // ячейка 61

  // ячейки 62–64: прямой поиск по резюме
  const summaryDocsOnly = await summariesCollection.similaritySearch("Путешествие по Корнуоллу");
  console.log(`\nDirect summary search: ${summaryDocsOnly.length} doc(s)`);
  console.log(summaryDocsOnly);
  // ячейка 65: ПРИМЕЧАНИЕ — прямой поиск по резюме получает более плотную информацию, но упускает детали.
}

// ─────────────────────────────────────────────────────────────────────────────
// ⑥  Мульти-векторный ретривер с гипотетическими вопросами (ячейки 66–77)
// ─────────────────────────────────────────────────────────────────────────────
async function runMultiVectorHypotheticalQuestions() {
  console.log("\n========== ⑥ Multi-Vector Retriever — Hypothetical Questions ==========");

  await resetChromaCollection("uk_hypothetical_questions"); // #C

  const hypotheticalQuestionsCollection = new Chroma(embeddings, { // #B
    collectionName: "uk_hypothetical_questions",
  });

  const docByteStore = new InMemoryStore<Uint8Array>(); // #D
  const docKey = "doc_id";

  const multiVectorRetriever = new MultiVectorRetriever({ // #E
    vectorstore: hypotheticalQuestionsCollection,
    byteStore: docByteStore,
    idKey: docKey,
  });

  // ячейка 69: LLM со структурированным выводом
  const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    openAIApiKey: OPENAI_API_KEY,
  });

  // ячейка 68: схема (Pydantic BaseModel → Zod)
  const HypotheticalQuestionsSchema = z.object({
    questions: z
      .array(z.string())
      .describe("Список гипотетических вопросов для данного текста"),
  });

  const llmWithStructuredOutput = llm.withStructuredOutput( // ячейка 69
    HypotheticalQuestionsSchema
  );

  // ячейка 70: цепочка гипотетических вопросов
  const hypotheticalQuestionsChain = ChatPromptTemplate.fromTemplate( // #B
    "Сгенерируй список ровно из 4 гипотетических вопросов, на которые мог бы ответить следующий текст:\n\n{document_text}"
  )
    .pipe(llmWithStructuredOutput) // #C
    .pipe((x: z.infer<typeof HypotheticalQuestionsSchema>) => x.questions); // #D

  // ячейка 71: загрузка с гипотетическими вопросами
  for (const destinationUrl of ukDestinationUrls) {
    const htmlDocs = await loadHtmlDocument(destinationUrl); // #A #B
    const textDocs = htmlToTextDocs(htmlDocs); // #C

    const coarseChunks = await parentSplitter.splitDocuments(textDocs); // #D
    const coarseChunkIds = coarseChunks.map(() => uuidv4());

    const allHypotheticalQuestions: Document[] = [];

    for (let i = 0; i < coarseChunks.length; i++) { // #E
      const coarseChunkId = coarseChunkIds[i];

      const hypotheticalQuestions = await hypotheticalQuestionsChain.invoke({ // #F
        document_text: coarseChunks[i].pageContent,
      });

      const questionDocs = hypotheticalQuestions.map( // #G
        (question: string) =>
          new Document({
            pageContent: question,
            metadata: { [docKey]: coarseChunkId },
          })
      );
      allHypotheticalQuestions.push(...questionDocs);
    }

    console.log(`Ingesting ${destinationUrl}`);
    await multiVectorRetriever.vectorstore.addDocuments(allHypotheticalQuestions); // #H
    await multiVectorRetriever.docstore.mset(
      coarseChunkIds.map((id, i) => [id, coarseChunks[i]]) // #I
    );
  }

  // ячейка 72: извлечение
  const retrievedDocs = await multiVectorRetriever.invoke(
    "Как добраться из Лондона в Брайтон?"
  );
  console.log(`\nRetrieved ${retrievedDocs.length} parent doc(s)`); // ячейка 73
  console.log(retrievedDocs); // ячейка 74

  // ячейки 75–77: прямой поиск по вопросам
  const questionDocsOnly = await hypotheticalQuestionsCollection.similaritySearch(
    "Как добраться из Лондона в Брайтон?"
  );
  console.log(`\nDirect question search: ${questionDocsOnly.length} doc(s)`);
  console.log(questionDocsOnly);
}

// ─────────────────────────────────────────────────────────────────────────────
// ⑦  Мульти-векторный ретривер с расширенными (оконными) чанками (ячейки 78–87)
// ─────────────────────────────────────────────────────────────────────────────
async function runMultiVectorExpandedContext() {
  console.log("\n========== ⑦ Multi-Vector Retriever — Expanded Context ==========");

  await resetChromaCollection("uk_granular_chunks"); // #C

  const granularChunksCollection = new Chroma(embeddings, { // #B
    collectionName: "uk_granular_chunks",
  });

  const expandedChunkStore = new InMemoryStore<Uint8Array>(); // #D
  const docKey = "doc_id";

  const multiVectorRetriever = new MultiVectorRetriever({ // #E
    vectorstore: granularChunksCollection,
    byteStore: expandedChunkStore,
    idKey: docKey,
  });

  // ячейка 80: загрузка с расширенным контекстом
  for (const destinationUrl of ukDestinationUrls) {
    const htmlDocs = await loadHtmlDocument(destinationUrl); // #A #B
    const textDocs = htmlToTextDocs(htmlDocs); // #C

    const granularChunks = await granularChunkSplitter.splitDocuments(textDocs); // #D

    const expandedChunkStoreItems: [string, Document][] = [];

    for (let i = 0; i < granularChunks.length; i++) { // #E
      const thisChunkNum = i; // #F
      const previousChunkNum = i === 0 ? null : i - 1; // #F
      const nextChunkNum = i === granularChunks.length - 1 ? null : i + 1; // #F

      let expandedChunkText = ""; // #G

      if (previousChunkNum !== null) {
        expandedChunkText += granularChunks[previousChunkNum].pageContent + "\n";
      }
      expandedChunkText += granularChunks[thisChunkNum].pageContent + "\n";
      if (nextChunkNum !== null) {
        expandedChunkText += granularChunks[nextChunkNum].pageContent + "\n";
      }

      const expandedChunkId = uuidv4(); // #H
      const expandedChunkDoc = new Document({ pageContent: expandedChunkText }); // #I

      expandedChunkStoreItems.push([expandedChunkId, expandedChunkDoc]);
      granularChunks[i].metadata[docKey] = expandedChunkId; // #J
    }

    console.log(`Ingesting ${destinationUrl}`);
    await multiVectorRetriever.vectorstore.addDocuments(granularChunks); // #K
    await multiVectorRetriever.docstore.mset(expandedChunkStoreItems); // #L
  }

  // ячейка 81: извлечение
  const retrievedDocs = await multiVectorRetriever.invoke("Корнуолл Рейнджер");
  console.log(`\nRetrieved ${retrievedDocs.length} expanded doc(s)`); // ячейка 82
  if (retrievedDocs.length > 0) console.log(retrievedDocs[0]); // ячейка 83
  // КОММЕНТАРИЙ: расширенный чанк содержит более полезный контекст.

  // ячейки 84–86: прямой поиск по дочерним чанкам
  const childDocsOnly = await granularChunksCollection.similaritySearch("Корнуолл Рейнджер");
  console.log(`\nDirect child search: ${childDocsOnly.length} chunk(s)`);
  if (childDocsOnly.length > 0) console.log(childDocsOnly[0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Главная функция — запуск всех стратегий последовательно
// ─────────────────────────────────────────────────────────────────────────────
async function main() {
  await runGranularVsCoarseSingleUrl();
  await runMultiUrlGranularVsCoarse();
  await runParentDocumentRetriever();
  await runMultiVectorChildChunks();
  await runMultiVectorSummaries();
  await runMultiVectorHypotheticalQuestions();
  await runMultiVectorExpandedContext();
}

main().catch(console.error);
