
import OpenAI from "openai";
import { createCollection, ingestDocuments, executeLlmPrompt, myChatbot, queryVectorDatabase } from "./utils";
import * as dotenv from 'dotenv';

dotenv.config();

// ── Main ──────────────────────────────────────────────────────────────────────
async function main(): Promise<void> {
  // Ingestion
  console.log("init main")
  const collection = await createCollection();
  console.log("init ingest documents")
  await ingestDocuments(collection);
  console.log("Documents ingested into ChromaDB.\n");

  // Q&A demo
  const demoQuery = "Сколько дорических храмов находится в Пестуме?";
  const demoResult = await queryVectorDatabase(collection, demoQuery);
  console.log("Vector DB result for demo query:\n", demoResult, "\n");


  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) throw new Error("GROQ_API_KEY not found in .env");

  const client = new OpenAI({
    apiKey,
    baseURL: "https://api.groq.com/openai/v1",
  });

  // Trick question — naive prompt (no guardrails)
  const trickQuestion = "Сколько колонн у трёх храмов вместе взятых?";
  const trickContext = await queryVectorDatabase(collection, trickQuestion);
  const naivePrompt = `Прочитай следующий текст и ответь на вопрос: ${trickQuestion}. \nТекст: ${trickContext}`;
  const naiveAnswer = await executeLlmPrompt(client, naivePrompt);
  console.log("Naive prompt — trick question answer:\n", naiveAnswer, "\n");

  // Trick question — safer prompt
  const safeAnswer = await myChatbot(collection, client, trickQuestion);
  console.log("Safer prompt — trick question answer:\n", safeAnswer, "\n");

  // Full chatbot question
  const question = "Сколько храмов в Пестуме, кто их построил и в каком архитектурном стиле?";
  const answer = await myChatbot(collection, client, question);
  console.log("Chatbot answer:\n", answer);
}

main().catch(console.error);

