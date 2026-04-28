/**
 * Секция 3: Генерация SQL-запросов
 * Ячейки 44–55: Инициализируем SQLite в памяти из SQL-файлов, используем Groq LLM
 * для генерации SQL из естественного языка, очищаем вывод и выполняем запрос.
 */

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
 * StringOutputParser — парсер, извлекающий plain-строку из ответа LLM.
 * Преобразует объект AIMessage в обычную строку через message.content.
 * Используется как последнее звено в цепочках, где нужен текстовый ответ.
 * Самый простой из парсеров LangChain.
 */
import { StringOutputParser } from "@langchain/core/output_parsers";

import { GROQ_API_KEY } from "../utils/config";
import { getDatabase, getTableSchemas, execQuery, cleanSql } from "../utils/dbUtils";

export async function runSqlQueryGeneration() {
  console.log("\n========== ③ Генерация SQL-запросов ==========");

  // Ячейка 44 — инициализируем базу данных в памяти
  const db = await getDatabase();
  const schema = getTableSchemas(db);
  console.log("Схема базы данных загружена.");

  const llm = new ChatOpenAI({
    modelName: "llama-3.3-70b-versatile",
    openAIApiKey: GROQ_API_KEY,
    configuration: { baseURL: "https://api.groq.com/openai/v1" },
  });

  // Ячейки 48–52 — цепочка генерации SQL с последующей очисткой вывода
  const sqlGenPrompt = ChatPromptTemplate.fromTemplate(
    `Given the following SQLite database schema, write a SQL SELECT query to answer the question.
Return ONLY the SQL query ending with a semicolon, no explanation or markdown.

Schema:
{schema}

Question: {question}
SQL:`
  );

  const sqlGenChain = sqlGenPrompt.pipe(llm).pipe(new StringOutputParser());

  // Ячейка 48 — тестовый запрос из ноутбука: предложения для Кардиффа
  const question = "Give me some offers for Cardiff, including the hotel name";
  console.log(`\nВопрос: "${question}"`);

  const rawSql = await sqlGenChain.invoke({ schema, question });
  console.log("\nСырой вывод LLM:\n", rawSql);

  // Ячейка 52 — очистка SQL (убираем markdown и преамбулу)
  const cleanedSql = cleanSql(rawSql);
  console.log("\nОчищенный SQL:\n", cleanedSql);

  // Ячейки 53–55 — выполняем запрос
  try {
    const results = execQuery(db, cleanedSql);
    console.log("\nРезультаты запроса:");
    console.log(JSON.stringify(results, null, 2));
  } catch (error) {
    console.error("Ошибка выполнения SQL:", error);
  }

  db.close();
}

async function main() {
  await runSqlQueryGeneration();
}

if (require.main === module) {
  main().catch(console.error);
}
