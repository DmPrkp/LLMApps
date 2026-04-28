/**
 * Секция 5: RAG Fusion
 * Ячейки 70–97: Генерируем 5 вариаций запроса, делаем поиск по каждой,
 * применяем алгоритм Reciprocal Rank Fusion (RRF) для переранжирования,
 * затем формируем ответ на основе топ-3 документов.
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

/*
 * Document — базовый класс для представления текстового документа в LangChain.
 * Содержит два поля: pageContent (текст) и metadata (произвольный объект с метаданными).
 * Используется на всех этапах пайплайна: загрузка, разбивка на чанки, хранение и retrieval.
 * Все векторные хранилища и retriever-ы принимают и возвращают массивы Document.
 */
import { Document } from "@langchain/core/documents";

import { embeddings, GROQ_API_KEY } from "../utils/config";
import { COLLECTION_NAME } from "./section1_ingest";

// Ячейки 78–89 — алгоритм Reciprocal Rank Fusion
// Каждый документ получает очки 1/(rank + k) из каждой группы результатов,
// суммарный скор определяет итоговый рейтинг
function reciprocalRankFusion(docGroups: Document[][], k = 60): Document[] {
  const scoreMap = new Map<string, { doc: Document; score: number }>();

  for (const group of docGroups) {
    group.forEach((doc, rank) => {
      const key = doc.pageContent;
      const entry = scoreMap.get(key);
      const increment = 1 / (rank + k);

      if (entry) {
        entry.score += increment;
      } else {
        scoreMap.set(key, { doc, score: increment });
      }
    });
  }

  return Array.from(scoreMap.values())
    .sort((a, b) => b.score - a.score)
    .map(({ doc }) => doc);
}

export async function runRagFusion() {
  console.log("\n========== ⑤ RAG Fusion ==========");

  const collection = new Chroma(embeddings, { collectionName: COLLECTION_NAME });

  const llm = new ChatOpenAI({
    modelName: "llama-3.3-70b-versatile",
    openAIApiKey: GROQ_API_KEY,
    configuration: { baseURL: "https://api.groq.com/openai/v1" },
  });

  // Ячейки 70–77 — генерация 5 вариаций исходного запроса
  const QuestionsSchema = z.object({
    questions: z.array(z.string()).describe("Список из 5 альтернативных вопросов"),
  });

  const multiQueryPrompt = ChatPromptTemplate.fromTemplate(
    `You are an AI language model assistant. Your task is to generate five
different search queries that cover the same intent as the provided input question.
The goal is to overcome potential limitations of distance-based similarity search.

Provide your response as a JSON object with a "questions" key containing a list of the five alternative questions.

Original question: {question}

Questions:`
  );

  const questionsParser = new JsonOutputParser<z.infer<typeof QuestionsSchema>>();
  const multiQueryChain = multiQueryPrompt.pipe(llm).pipe(questionsParser);

  const userQuestion = "Can you give me some tips for a trip to Brighton?";
  console.log(`\nВопрос: "${userQuestion}"`);

  // Генерируем 5 вариаций
  const { questions } = await multiQueryChain.invoke({ question: userQuestion });
  console.log("\nСгенерированные вариации запроса:");
  questions.forEach((q, i) => console.log(`  ${i + 1}. ${q}`));

  // Ячейки 78–89 — ищем документы по каждой вариации и собираем группы
  const retriever = collection.asRetriever(4);
  const docGroups: Document[][] = [];

  for (const q of questions) {
    const docs = await retriever.invoke(q);
    docGroups.push(docs);
  }

  // Применяем RRF для объединения и переранжирования результатов
  const fusedDocs = reciprocalRankFusion(docGroups);
  const topDocs = fusedDocs.slice(0, 3);

  console.log(
    `\nRRF: объединено ${docGroups.flat().length} документов → топ-3:`
  );
  for (const doc of topDocs) {
    console.log(
      `  [${doc.metadata?.destination ?? "?"}/${doc.metadata?.region ?? "?"}] ${doc.pageContent.substring(0, 80)}...`
    );
  }

  // Ячейки 90–97 — финальный RAG-ответ на основе топ-3 документов
  const ragPrompt = ChatPromptTemplate.fromTemplate(
    `Given a question and some context, answer the question.
Only use the provided context to answer the question.
If you do not know the answer, just say I do not know.

Context: {context}
Question: {question}`
  );

  const context = topDocs.map((d) => d.pageContent).join("\n\n");

  const ragAnswer = await ragPrompt
    .pipe(llm)
    .pipe(new StringOutputParser())
    .invoke({ context, question: userQuestion });

  console.log("\nОтвет RAG Fusion:", ragAnswer);
}

async function main() {
  await runRagFusion();
}

if (require.main === module) {
  main().catch(console.error);
}
