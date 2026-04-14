/**
 * Section 3: Multi-Query Generation
 * Cells 18–28: Generate 5 different query variations from a single user question
 * to overcome limitations of distance-based similarity search.
 */
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { JsonOutputParser } from "@langchain/core/output_parsers";
import { MultiQueryRetriever } from "langchain/retrievers/multi_query";
import { z } from "zod";

import { embeddings, GROQ_API_KEY } from "../utils/config";

// Cell 18 — Zod schema (equivalent to Pydantic BaseModel + JsonOutputParser)
const QuestionsSchema = z.object({
  questions: z.array(z.string()).describe("List of questions"),
});
type QuestionsOutput = z.infer<typeof QuestionsSchema>;

async function main() {
  console.log("=== Section 3: Multi-Query Generation ===\n");

  const ukGranularCollection = new Chroma(embeddings, {
    collectionName: "uk_granular",
  });

  // Cell 20 — multi-query generation prompt
  const multiQueryGenPromptTemplate = `You are an AI language model assistant. Your task is to generate five
different search queries that cover the same intent as the provided input question.
The goal is to overcome potential limitations of distance-based similarity search.

Provide your response as a JSON object with a "questions" key containing a list of the five alternative questions.

Original question: {question}

Questions:`;

  const multiQueryGenPrompt = ChatPromptTemplate.fromTemplate(
    multiQueryGenPromptTemplate
  );

  // Cell 21 — LLM
  const llm = new ChatOpenAI({
    modelName: "llama-3.3-70b-versatile",
    openAIApiKey: GROQ_API_KEY,
    configuration: { baseURL: "https://api.groq.com/openai/v1" },
  });

  // Cell 22 — multi-query chain: prompt | llm | json parser
  const questionsParser = new JsonOutputParser<QuestionsOutput>();
  const multiQueryGenChain = multiQueryGenPrompt.pipe(llm).pipe(questionsParser);

  const userQuestion = "Tell me some fun things I can do in Cornwall";

  // Cell 23 — generate 5 alternative queries
  const multipleQueries = await multiQueryGenChain.invoke({
    question: userQuestion,
  });

  // Cell 24 — display results
  console.log("Generated queries:", multipleQueries);

  // Cells 25–26 — LangChain MultiQueryRetriever (standard implementation)
  const retriever = ukGranularCollection.asRetriever();
  const multiQueryRetriever = MultiQueryRetriever.fromLLM({
    retriever,
    llm,
  });

  // Cell 27 — retrieve using multi-query
  const multiQueryResults = await multiQueryRetriever.invoke(userQuestion);

  // Cell 28 — display results
  console.log(`\nMultiQueryRetriever returned ${multiQueryResults.length} unique docs:`);
  for (const doc of multiQueryResults) console.log(doc);
}

main().catch(console.error);
