import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { JsonOutputParser } from "@langchain/core/output_parsers";
import { RunnablePassthrough } from "@langchain/core/runnables";
import { z } from "zod";
import { MultiQueryRetriever } from "langchain/retrievers/multi_query";

import { embeddings, GROQ_API_KEY, ukDestinationUrls } from "./utils/config";
import {
  loadHtmlDocument,
  splitDocsIntoGranularChunks,
  resetChromaCollection,
} from "./utils/htmlUtils";

// ─────────────────────────────────────────────────────────────────────────────
// ①  Data ingestion — load HTML and populate Chroma (cells 1–8)
// ─────────────────────────────────────────────────────────────────────────────
async function runDataIngestion() {
  console.log("\n========== ① Data Ingestion ==========");

  // Cell 2 — create/reset collection
  await resetChromaCollection("uk_granular");

  const ukGranularCollection = new Chroma(embeddings, {
    collectionName: "uk_granular",
  });

  // Cell 8 — load each URL and split into granular H2 chunks
  for (const destinationUrl of ukDestinationUrls) {
    const docs = await loadHtmlDocument(destinationUrl); // cells 6–7
    console.log(`Loading ${destinationUrl}`);
    const granularChunks = splitDocsIntoGranularChunks(docs); // cells 6–8 #B–#D
    await ukGranularCollection.addDocuments(granularChunks);
  }

  console.log("Ingestion complete.");
}

// ─────────────────────────────────────────────────────────────────────────────
// ②  Rewrite-Retrieve-Read — rewrite question before vector search (cells 9–17)
// ─────────────────────────────────────────────────────────────────────────────
async function runQueryRewriter() {
  console.log("\n========== ② Rewrite-Retrieve-Read ==========");

  const ukGranularCollection = new Chroma(embeddings, {
    collectionName: "uk_granular",
  });

  const userQuestion = "Tell me some fun things I can enjoy in Cornwall";

  // Cell 9 — retrieve with original question (baseline, usually poor)
  console.log("\n--- Original question retrieval ---");
  const initialResults = await ukGranularCollection.similaritySearch(
    userQuestion,
    4
  );
  for (const doc of initialResults) console.log(doc);

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

  // Cell 14 — rewriter chain
  const rewriterChain = rewriterPrompt.pipe(llm).pipe(new StringOutputParser());

  // Cell 15 — test rewriter
  const searchQuery = await rewriterChain.invoke({ user_question: userQuestion });
  console.log("\nRewritten search query:", searchQuery); // leisure activities in Cornwall

  // Cell 16 — retrieve with rewritten query
  console.log("\n--- Rewritten query retrieval ---");
  const retriever = ukGranularCollection.asRetriever();
  const context = await retriever.invoke(searchQuery);

  // Cell 17 — final RAG answer
  const ragPromptTemplate = `Given a question and some context, answer the question.
If you do not know the answer, just say I do not know.

Context: {context}
Question: {question}`;

  const ragPrompt = ChatPromptTemplate.fromTemplate(ragPromptTemplate);

  const ragAnswer = await ragPrompt
    .pipe(llm)
    .pipe(new StringOutputParser())
    .invoke({ context, question: userQuestion });

  console.log("\nRewrite-Retrieve-Read answer: \n", ragAnswer);
}

// ─────────────────────────────────────────────────────────────────────────────
// ③  Multi-Query — generate 5 query variations for better coverage (cells 18–28)
// ─────────────────────────────────────────────────────────────────────────────
async function runMultiQuery() {
  console.log("\n========== ③ Multi-Query Generation ==========");

  const ukGranularCollection = new Chroma(embeddings, {
    collectionName: "uk_granular",
  });

  // Cell 18 — Zod schema (equivalent to Pydantic BaseModel)
  const QuestionsSchema = z.object({
    questions: z.array(z.string()).describe("List of questions"),
  });

  // Cell 20 — multi-query prompt
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

  // Cell 22 — multi-query chain with JSON output parser
  const questionsParser = new JsonOutputParser<z.infer<typeof QuestionsSchema>>();
  const multiQueryGenChain = multiQueryGenPrompt.pipe(llm).pipe(questionsParser);

  const userQuestion = "Tell me some fun things I can do in Cornwall";

  // Cell 23 — generate multiple queries
  const multipleQueries = await multiQueryGenChain.invoke({ question: userQuestion });
  console.log("\nGenerated queries:", multipleQueries);

  // Cells 25–28 — use LangChain MultiQueryRetriever
  const retriever = ukGranularCollection.asRetriever();
  const multiQueryRetriever = MultiQueryRetriever.fromLLM({
    retriever,
    llm,
  });

  const multiQueryResults = await multiQueryRetriever.invoke(userQuestion);
  console.log(`\nMultiQueryRetriever returned ${multiQueryResults.length} docs`);
  for (const doc of multiQueryResults) console.log(doc);
}

// ─────────────────────────────────────────────────────────────────────────────
// ④  Step-back Questions — generate broader questions for wider context (cells 29–38)
// ─────────────────────────────────────────────────────────────────────────────
async function runStepBack() {
  console.log("\n========== ④ Step-back Questions ==========");

  const ukGranularCollection = new Chroma(embeddings, {
    collectionName: "uk_granular",
  });

  // Cell 29 — step-back prompt
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
  console.log("\nStep-back question:", stepBackQuestion);

  // Cell 34 — retrieve using step-back question
  console.log("\n--- Step-back retrieval ---");
  const stepBackResults = await ukGranularCollection.similaritySearch(
    stepBackQuestion,
    4
  );
  for (const doc of stepBackResults) console.log(doc);

  // Cells 35–38 — full step-back RAG chain
  const retriever = ukGranularCollection.asRetriever();

  const ragPromptTemplate = `Given a question and some context, answer the question.
If you do not know the answer, just say I do not know.

Context: {context}
Question: {question}`;

  const ragPrompt = ChatPromptTemplate.fromTemplate(ragPromptTemplate);

  // #A context: original question → step-back chain → retriever
  // #B question: original question passed through
  const stepBackRagChain = {
    context: new RunnablePassthrough().pipe(stepBackChain).pipe(retriever),
    question: new RunnablePassthrough(),
  } as any;

  const stepBackRagAnswer = await ragPrompt
    .pipe(llm)
    .pipe(new StringOutputParser())
    .invoke(
      await (async () => {
        const stepBack = await stepBackChain.invoke({ question: userQuestion });
        const context = await retriever.invoke(stepBack);
        return { context, question: userQuestion };
      })()
    );

  console.log("\nStep-back RAG answer:", stepBackRagAnswer);
}

// ─────────────────────────────────────────────────────────────────────────────
// ⑤  HyDE — Hypothetical Document Embeddings (cells 39–45)
// ─────────────────────────────────────────────────────────────────────────────
async function runHyDE() {
  console.log("\n========== ⑤ HyDE (Hypothetical Document Embeddings) ==========");

  const ukGranularCollection = new Chroma(embeddings, {
    collectionName: "uk_granular",
  });

  // Cell 39 — LLM
  const llm = new ChatOpenAI({
    modelName: "llama-3.3-70b-versatile",
    openAIApiKey: GROQ_API_KEY,
    configuration: { baseURL: "https://api.groq.com/openai/v1" },
  });

  // Cell 40 — HyDE prompt
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
  console.log("\nHypothetical document:", hypotheticalDocument);

  // Cell 44 — HyDE RAG chain
  const retriever = ukGranularCollection.asRetriever();

  const ragPromptTemplate = `Given a question and some context, answer the question.
Only use the provided context to answer the question.
If you do not know the answer, just say I do not know.

Context: {context}
Question: {question}`;

  const ragPrompt = ChatPromptTemplate.fromTemplate(ragPromptTemplate);

  // #A context: question → HyDE chain → retriever
  // #B question: original question passed through
  const hydeAnswer = await ragPrompt
    .pipe(llm)
    .pipe(new StringOutputParser())
    .invoke(
      await (async () => {
        const hypothetical = await hydeChain.invoke({ question: userQuestion });
        const context = await retriever.invoke(hypothetical);
        return { context, question: userQuestion };
      })()
    );

  // Cell 45
  console.log("\nHyDE RAG answer:", hydeAnswer);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main — run all sections sequentially
// ─────────────────────────────────────────────────────────────────────────────
async function main() {
  // Uncomment to re-ingest data (only needed once or when resetting):
  // await runDataIngestion();

  await runQueryRewriter();
  // await runMultiQuery();
  // await runStepBack();
  // await runHyDE();
}

main().catch(console.error);
