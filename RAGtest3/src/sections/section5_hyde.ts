/**
 * Section 5: HyDE — Hypothetical Document Embeddings
 * Cells 39–45: Generate a hypothetical sentence that would answer the question,
 * use it for embedding-based retrieval, then answer the original question.
 */
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

import { embeddings, GROQ_API_KEY } from "../utils/config";

async function main() {
  console.log("=== Section 5: HyDE (Hypothetical Document Embeddings) ===\n");

  const ukGranularCollection = new Chroma(embeddings, {
    collectionName: "uk_granular",
  });

  // Cell 39 — LLM
  const llm = new ChatOpenAI({
    modelName: "llama-3.3-70b-versatile",
    openAIApiKey: GROQ_API_KEY,
    configuration: { baseURL: "https://api.groq.com/openai/v1" },
  });

  // Cell 40 — HyDE prompt: generate a single sentence that could answer the question
  const hydePromptTemplate = `Write one sentence that could answer the provided question.
Do not add anything else.
Question: {question}
Sentence:`;

  const hydePrompt = ChatPromptTemplate.fromTemplate(hydePromptTemplate);

  // Cell 41 — HyDE chain
  const hydeChain = hydePrompt.pipe(llm).pipe(new StringOutputParser());

  const userQuestion = "What are the best beaches in Cornwall?";

  // Cell 42 — generate hypothetical document
  const hypotheticalDocument = await hydeChain.invoke({ question: userQuestion });

  // Cell 43 — display hypothetical document
  console.log("Hypothetical document:", hypotheticalDocument);

  // Cell 44 — HyDE RAG chain
  // #A context: question → HyDE chain → retriever (use hypothetical doc for search)
  // #B question: original question passed through
  const retriever = ukGranularCollection.asRetriever();

  const ragPromptTemplate = `Given a question and some context, answer the question.
Only use the provided context to answer the question.
If you do not know the answer, just say I do not know.

Context: {context}
Question: {question}`;

  const ragPrompt = ChatPromptTemplate.fromTemplate(ragPromptTemplate);

  const hypothetical = await hydeChain.invoke({ question: userQuestion }); // #A
  const context = await retriever.invoke(hypothetical); // #A

  const answer = await ragPrompt
    .pipe(llm)
    .pipe(new StringOutputParser())
    .invoke({ context, question: userQuestion }); // #B

  // Cell 45
  console.log("\nHyDE RAG answer:", answer);
}

main().catch(console.error);
