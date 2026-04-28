/**
 * Секция 4: Роутер запросов
 * Ячейки 56–69: LLM определяет, куда направить вопрос пользователя —
 * в векторное хранилище Chroma (туристическая информация)
 * или в SQLite базу бронирований (предложения и скидки).
 */
import { z } from "zod";
import { Database } from "sql.js";

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
import { getDatabase, getTableSchemas, execQuery, cleanSql } from "../utils/dbUtils";
import { COLLECTION_NAME } from "./section1_ingest";

// Ячейки 56–57 — схема RouteQuery: LLM выбирает один из двух источников данных
const RouteQuerySchema = z.object({
  datasource: z
    .enum(["tourist_info_store", "uk_booking_db"])
    .describe(
      "Route to 'uk_booking_db' for accommodation offers, bookings, prices, deals. " +
        "Route to 'tourist_info_store' for travel tips, attractions, and sightseeing."
    ),
});

type RouteQuery = z.infer<typeof RouteQuerySchema>;

export async function runQueryRouter() {
  console.log("\n========== ④ Роутер запросов ==========");

  const collection = new Chroma(embeddings, { collectionName: COLLECTION_NAME });
  const db: Database = await getDatabase();
  const schema = getTableSchemas(db);

  const llm = new ChatOpenAI({
    modelName: "llama-3.3-70b-versatile",
    openAIApiKey: GROQ_API_KEY,
    configuration: { baseURL: "https://api.groq.com/openai/v1" },
  });

  // Ячейки 58–60 — цепочка роутера с JSON-выводом
  const routerPrompt = ChatPromptTemplate.fromTemplate(
    `You are an expert at routing user questions to the right data source.
Route to "uk_booking_db" for questions about accommodation offers, bookings, deals, or prices.
Route to "tourist_info_store" for questions about travel tips, attractions, sightseeing, or local info.

Return ONLY a JSON object: {{"datasource": "<choice>"}}

Question: {question}
JSON:`
  );

  const routerParser = new JsonOutputParser<RouteQuery>();
  const routerChain = routerPrompt.pipe(llm).pipe(routerParser);

  // Цепочка генерации SQL для случая маршрутизации в базу бронирований
  const sqlGenPrompt = ChatPromptTemplate.fromTemplate(
    `Given the following SQLite database schema, write a SQL SELECT query to answer the question.
Return ONLY the SQL query ending with a semicolon, no explanation or markdown.

Schema:
{schema}

Question: {question}
SQL:`
  );
  const sqlGenChain = sqlGenPrompt.pipe(llm).pipe(new StringOutputParser());

  // Общий RAG-промпт для формирования финального ответа
  const ragPrompt = ChatPromptTemplate.fromTemplate(
    `Given a question and some context, answer the question.
If you do not know the answer, just say I do not know.

Context: {context}
Question: {question}`
  );

  // Ячейки 63–65 — функция выбора и вызова нужной цепочки
  async function executeRoutedQuery(question: string) {
    console.log(`\nВопрос: "${question}"`);

    const route = await routerChain.invoke({ question });
    console.log(`Направлено в: ${route.datasource}`);

    let context: string;

    if (route.datasource === "uk_booking_db") {
      // Путь через SQL: генерируем запрос и выполняем в SQLite
      const rawSql = await sqlGenChain.invoke({ schema, question });
      const cleanedSql = cleanSql(rawSql);
      console.log("Сгенерированный SQL:", cleanedSql);
      const results = execQuery(db, cleanedSql);
      context = JSON.stringify(results, null, 2);
    } else {
      // Путь через векторный поиск в Chroma
      const retriever = collection.asRetriever(4);
      const docs = await retriever.invoke(question);
      context = docs.map((d) => d.pageContent).join("\n\n");
    }

    const answer = await ragPrompt
      .pipe(llm)
      .pipe(new StringOutputParser())
      .invoke({ context, question });

    console.log("Ответ:", answer);
  }

  // Ячейки 66–69 — тестовые запросы из ноутбука
  await executeRoutedQuery("Have you got any offers in Brighton?");
  await executeRoutedQuery("Where are the best beaches in Cornwall?");

  db.close();
}

async function main() {
  await runQueryRouter();
}

if (require.main === module) {
  main().catch(console.error);
}
