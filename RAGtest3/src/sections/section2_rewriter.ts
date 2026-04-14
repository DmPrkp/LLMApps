/**
 * Section 2: Rewrite-Retrieve-Read
 * Cells 9–17: Rewrite user question into a better vector DB query
 * to improve semantic search results.
 */
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

import { embeddings, GROQ_API_KEY } from "../utils/config";

async function main() {
  console.log("=== Section 2: Rewrite-Retrieve-Read ===\n");

  const ukGranularCollection = new Chroma(embeddings, {
    collectionName: "uk_granular",
  });

  const userQuestion = "Tell me some fun things I can enjoy in Cornwall";

  // Cell 9 — baseline retrieval with original question (usually poor results)
  console.log("--- Original question retrieval ---");
  const initialResults = await ukGranularCollection.similaritySearch(
    userQuestion,
    4
  );
  for (const doc of initialResults) console.log(doc);
  // COMMENT: retrieval from vector store against original question is bad

  // Cells 11–13 — rewriter prompt
  const rewriterPromptTemplate = `Generate search query for the Chroma DB vector store
from a user question, allowing for a more accurate
response through semantic search.
Just return the revised Chroma DB query, with quotes around it.

User question: {user_question}
Revised Chroma DB query:`;

  const rewriterPrompt = ChatPromptTemplate.fromTemplate(rewriterPromptTemplate);

  // Cell 12 — LLM
  const llm = new ChatOpenAI({
    modelName: "llama-3.3-70b-versatile",
    openAIApiKey: GROQ_API_KEY,
    configuration: { baseURL: "https://api.groq.com/openai/v1" },
  });

  // Cell 14 — rewriter chain: prompt | llm | string parser
  const rewriterChain = rewriterPrompt.pipe(llm).pipe(new StringOutputParser());

  // Cell 15 — invoke rewriter
  const searchQuery = await rewriterChain.invoke({ user_question: userQuestion });
  console.log("\nRewritten search query:", searchQuery);

  // Cell 16 — retrieve with rewritten query
  console.log("\n--- Rewritten query retrieval ---");
  const revisedResults = await ukGranularCollection.similaritySearch(
    searchQuery,
    4
  );
  for (const doc of revisedResults) console.log(doc);
  // COMMENT: retrieval with rewritten question is much better!
}

main().catch(console.error);
