/**
 * Секция 2: Фильтрация по метаданным и генерация структурированных запросов
 * Ячейки 18–43: Ручная фильтрация по метаданным, затем LLM-генерация
 * структурированного запроса через Zod-схему + JSON-вывод для построения фильтров Chroma.
 */
import { z } from "zod";

/*
 * Chroma — интеграция с векторным хранилищем ChromaDB.
 * Позволяет сохранять документы как векторные эмбеддинги и искать по семантическому сходству.
 * Поддерживает фильтрацию по метаданным через операторы $eq, $and, $or.
 * Используется как основное хранилище для RAG-поиска во всех секциях.
 */
import { Chroma } from "@langchain/community/vectorstores/chroma";

/*
 * ChatOpenAI — обёртка над OpenAI Chat API и совместимыми API (в т.ч. Groq).
 * Позволяет настраивать модель, температуру и baseURL для кастомных провайдеров.
 * Поддерживает вызов инструментов, структурированный вывод и стриминг ответов.
 * Здесь используется с Groq API через OpenAI-совместимый endpoint.
 */
import { ChatOpenAI } from "@langchain/openai";

/*
 * ChatPromptTemplate — шаблон для формирования сообщений чату с переменными-подстановками.
 * fromTemplate() создаёт промпт с плейсхолдерами вида {variable}, которые подставляются при вызове.
 * Является первым звеном в LCEL-цепочках (LangChain Expression Language).
 * Поддерживает системные, пользовательские и AI-сообщения в одном шаблоне.
 */
import { ChatPromptTemplate } from "@langchain/core/prompts";

/*
 * JsonOutputParser — парсер, извлекающий JSON-объект из текстового ответа LLM.
 * Автоматически находит JSON в тексте даже если модель добавляет лишний текст вокруг него.
 * Типизируется через дженерик: JsonOutputParser<MyType> для типобезопасного вывода.
 * Используется для структурированного вывода без function calling.
 *
 * StringOutputParser — парсер, извлекающий plain-строку из ответа LLM.
 * Преобразует объект AIMessage в обычную строку через message.content.
 * Используется как последнее звено в цепочках, где нужен текстовый ответ.
 * Самый простой из парсеров LangChain.
 */
import { JsonOutputParser, StringOutputParser } from "@langchain/core/output_parsers";

import { embeddings, GROQ_API_KEY } from "../utils/config";
import { COLLECTION_NAME } from "./section1_ingest";

// Ячейки 30–31 — Zod-схема, аналог Pydantic-модели DestinationSearch из ноутбука
const DestinationSearchSchema = z.object({
  content_search: z
    .string()
    .describe("Тема или контент для поиска в векторном хранилище"),
  destination: z
    .string()
    .describe("Конкретное UK-направление, например Newquay"),
  region: z
    .string()
    .describe("Регион: Cornwall или East Sussex"),
});

type DestinationSearch = z.infer<typeof DestinationSearchSchema>;

// Ячейка 32 — строим Chroma-фильтр $and/$eq из полей DestinationSearch
function buildFilter(search: DestinationSearch): object | undefined {
  const conditions: object[] = [];

  if (search.destination) {
    conditions.push({ destination: { $eq: search.destination } });
  }
  if (search.region) {
    conditions.push({ region: { $eq: search.region } });
  }

  if (conditions.length === 0) return undefined;
  if (conditions.length === 1) return conditions[0];
  return { $and: conditions };
}

export async function runMetadataFiltering() {
  console.log("\n========== ② Фильтрация по метаданным и структурированный запрос ==========");

  const collection = new Chroma(embeddings, { collectionName: COLLECTION_NAME });

  const llm = new ChatOpenAI({
    modelName: "llama-3.3-70b-versatile",
    openAIApiKey: GROQ_API_KEY,
    configuration: { baseURL: "https://api.groq.com/openai/v1" },
  });

  // Ячейки 18–19 — ручная фильтрация по метаданным
  console.log("\n--- А) Ручная фильтрация по метаданным ---");
  const manualResults = await collection.similaritySearch(
    "events or festivals",
    4,
    { destination: "Newquay" }
  );
  console.log(`Ручной фильтр вернул ${manualResults.length} документов:`);
  for (const doc of manualResults) {
    console.log(
      `  [${doc.metadata.destination} / ${doc.metadata.region}] ${doc.pageContent.substring(0, 80)}...`
    );
  }

  // Ячейки 30–43 — генерация структурированного запроса через JSON-вывод LLM
  console.log("\n--- Б) Генерация структурированного запроса ---");

  const queryGenPrompt = ChatPromptTemplate.fromTemplate(
    `Extract destination search parameters from the user question about UK travel.
Return a JSON object with exactly these fields:
- content_search: the topic to search for (e.g. "events and festivals")
- destination: the specific destination name with spaces (e.g. "Newquay", "St Ives")
- region: either "Cornwall" or "East Sussex"

Return ONLY the JSON object, no explanation.

Question: {question}
JSON:`
  );

  const parser = new JsonOutputParser<DestinationSearch>();
  const queryGenChain = queryGenPrompt.pipe(llm).pipe(parser);

  const userQuestion = "Tell me about events or festivals in the UK town of Newquay";
  console.log(`\nВопрос: "${userQuestion}"`);

  const structuredQuery = await queryGenChain.invoke({ question: userQuestion });
  console.log("\nСтруктурированный запрос:", structuredQuery);

  const filter = buildFilter(structuredQuery);
  console.log("Фильтр Chroma:", JSON.stringify(filter, null, 2));

  const structuredResults = await collection.similaritySearch(
    structuredQuery.content_search,
    4,
    filter
  );
  console.log(`\nСтруктурированный запрос вернул ${structuredResults.length} документов:`);
  for (const doc of structuredResults) {
    console.log(
      `  [${doc.metadata.destination} / ${doc.metadata.region}] ${doc.pageContent.substring(0, 80)}...`
    );
  }

  // Финальный RAG-ответ на основе найденного контекста
  const ragPrompt = ChatPromptTemplate.fromTemplate(
    `Given a question and some context, answer the question.
If you do not know the answer, just say I do not know.

Context: {context}
Question: {question}`
  );

  const ragAnswer = await ragPrompt
    .pipe(llm)
    .pipe(new StringOutputParser())
    .invoke({ context: structuredResults, question: userQuestion });

  console.log("\nRAG-ответ:", ragAnswer);
}

async function main() {
  await runMetadataFiltering();
}

if (require.main === module) {
  main().catch(console.error);
}
