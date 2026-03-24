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
    `Используйте следующие фрагменты контекста, чтобы ответить на вопрос в конце.

Если вы не знаете ответа, просто скажите, что не знаете, не пытайтесь придумать ответ.
Используйте максимум три предложения и постарайтесь ответить как можно короче.

{context}
Вопрос: {question}
Полезный ответ:`
  );

  const retriever = vectorDb.asRetriever();
  const chatbot = new ChatOpenAI({
    openAIApiKey: process.env.GROQ_API_KEY,
    modelName: "llama-3.3-70b-versatile",
    configuration: { baseURL: "https://api.groq.com/openai/v1" },
  });

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
    ["system", "Ты полезный помощник, эксперт мирового уровня по римской и греческой истории, особенно городов юга Италии. Делись интересными фактами о местной истории и рекомендуй места для посещения, давая увлекательные и глубокие ответы. Отвечай на все вопросы максимально полно, но используй только то, что предоставлено в контексте. Если не знаешь — так и скажи. Используй максимум три предложения и отвечай как можно короче."],
    ["placeholder", "{chat_history_messages}"],
    ["assistant", "{retrieved_context}"],
    ["human", "{question}"],
  ]);

  const retriever = vectorDb.asRetriever();
  const chatbot = new ChatOpenAI({
    openAIApiKey: process.env.GROQ_API_KEY,
    modelName: "llama-3.3-70b-versatile",
    configuration: { baseURL: "https://api.groq.com/openai/v1" },
  });
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
