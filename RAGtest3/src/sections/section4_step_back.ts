/**
 * Section 4: Step-back Questions
 * Cells 29–38: Generate a more general "step-back" question from a specific one
 * to retrieve broader context, then answer the original question.
 */
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

import { embeddings, GROQ_API_KEY } from "../utils/config";

async function main() {
  console.log("=== Section 4: Step-back Questions ===\n");

  const ukGranularCollection = new Chroma(embeddings, {
    collectionName: "uk_granular",
  });

  // Cell 29 — step-back question prompt
  const stepBackPromptTemplate = `You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more general, step-back question that is easier to answer.

Here are a few examples:

Original question: Could the members of The Police perform together after Sting's departure?
Stepback question: When did the members of the Police split up and was there any reunion?

Original question: How can I improve my time on 100m dash?
Stepback question: What are the ways to improve short distance sprinting performance?

Original question: What are some strategies to avoid long-term joint pain?
Stepback question: What are common causes of joint pain and how to treat them?

Original question: {question}
Stepback question:`;

  const stepBackPrompt = ChatPromptTemplate.fromTemplate(stepBackPromptTemplate);

  // Cell 30 — LLM
  const llm = new ChatOpenAI({
    modelName: "llama-3.3-70b-versatile",
    openAIApiKey: GROQ_API_KEY,
    configuration: { baseURL: "https://api.groq.com/openai/v1" },
  });

  // Cell 31 — step-back chain
  const stepBackChain = stepBackPrompt.pipe(llm).pipe(new StringOutputParser());

  const userQuestion = "Can you give me some tips for a trip to Brighton?";

  // Cell 32 — generate step-back question
  const stepBackQuestion = await stepBackChain.invoke({ question: userQuestion });

  // Cell 33 — display step-back question
  console.log("Step-back question:", stepBackQuestion);

  // Cell 34 — retrieve using step-back question
  console.log("\n--- Step-back retrieval ---");
  const stepBackResults = await ukGranularCollection.similaritySearch(
    stepBackQuestion,
    4
  );
  for (const doc of stepBackResults) console.log(doc);

  // Cells 35–38 — full step-back RAG chain
  // #A context: original question → step-back → retriever
  // #B question: original question passed through
  const retriever = ukGranularCollection.asRetriever();

  const ragPromptTemplate = `Given a question and some context, answer the question.
If you do not know the answer, just say I do not know.

Context: {context}
Question: {question}`;

  const ragPrompt = ChatPromptTemplate.fromTemplate(ragPromptTemplate);

  const stepBackContextQuestion = await stepBackChain.invoke({
    question: userQuestion,
  });
  const context = await retriever.invoke(stepBackContextQuestion); // #A

  const answer = await ragPrompt
    .pipe(llm)
    .pipe(new StringOutputParser())
    .invoke({ context, question: userQuestion }); // #B

  // Cell 38
  console.log("\nStep-back RAG answer:", answer);
}

main().catch(console.error);
