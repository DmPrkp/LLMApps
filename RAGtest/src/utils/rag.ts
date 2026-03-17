import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnablePassthrough, RunnableLambda } from "@langchain/core/runnables";
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

import { vectorDb } from "./config";

export async function similaritySearch(query: string, k = 4): Promise<void> {
  console.log(`\n=== Similarity search: "${query}" ===`);
  const results = await vectorDb.similaritySearch(query, k);
  console.log(`Found ${results.length} results:`);
  results.forEach((doc, i) => {
    console.log(`\n[${i + 1}] source: ${doc.metadata.source}`);
    console.log(doc.pageContent.slice(0, 200) + "...");
  });
}

export async function buildRagChain() {
  const ragPrompt = ChatPromptTemplate.fromTemplate(
    `Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:`
  );

  const retriever = vectorDb.asRetriever();
  const chatbot = new ChatOpenAI({ openAIApiKey: process.env.OPENAI_API_KEY, modelName: "gpt-4o-mini" });

  return RunnablePassthrough.assign({
    context: (input: { question: string }) => retriever.invoke(input.question),
  })
    .pipe(ragPrompt)
    .pipe(chatbot);
}

export async function executeChain(
  chain: Awaited<ReturnType<typeof buildRagChain>>,
  question: string
): Promise<string> {
  const answer = await chain.invoke({ question });
  return answer.content as string;
}

export async function buildRagChainWithMemory() {
  const ragPrompt = ChatPromptTemplate.fromMessages([
    ["system", "You are a helpful assistant, world-class expert in Roman and Greek history, especially in towns located in southern Italy. Provide interesting insights on local history and recommend places to visit with knowledgeable and engaging answers. Answer all questions to the best of your ability, but only use what has been provided in the context. If you don't know, just say you don't know. Use three sentences maximum and keep the answer as concise as possible."],
    ["placeholder", "{chat_history_messages}"],
    ["assistant", "{retrieved_context}"],
    ["human", "{question}"],
  ]);

  const retriever = vectorDb.asRetriever();
  const chatbot = new ChatOpenAI({ openAIApiKey: process.env.OPENAI_API_KEY, modelName: "gpt-4o-mini" });
  const chatHistoryMemory = new InMemoryChatMessageHistory();

  const ragChain = RunnablePassthrough.assign({
    retrieved_context: (input: { question: string }) => retriever.invoke(input.question),
    chat_history_messages: new RunnableLambda({ func: async () => chatHistoryMemory.getMessages() }),
  })
    .pipe(ragPrompt)
    .pipe(chatbot);

  return async function executeChainWithMemory(question: string): Promise<string> {
    await chatHistoryMemory.addMessage(new HumanMessage(question));
    const answer = await ragChain.invoke({ question });
    await chatHistoryMemory.addMessage(new AIMessage(answer.content as string));
    const messages = await chatHistoryMemory.getMessages();
    console.log(`\nFull chat history (${messages.length} messages)`);
    return answer.content as string;
  };
}
