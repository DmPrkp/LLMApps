/**
 * Секция 8: Checkpointer (память графа) + перемотка состояния.
 * Эквивалент main_08_01.py / main_08_02.py — компилируем граф с MemorySaver, прокидываем
 * thread_id в config, благодаря чему граф запоминает историю сообщений между invoke().
 * Часть 08_02 показывает get_state_history + откат к предыдущему checkpoint_id.
 */
import * as readline from "readline";
import { v1 as uuidv1 } from "uuid";
import { z } from "zod";

import {
  StateGraph,
  END,
  MessagesAnnotation,
  MemorySaver,
  Command,
} from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

import { getLlm } from "../utils/config";
import { searchTravelInfoTool } from "../utils/vectorstore";
import { weatherForecastTool } from "../utils/weather";
import { checkBnbAvailabilityTool } from "../utils/bnb";
import { getHotelDbToolkitTools } from "../utils/hotelDb";

const AgentType = {
  travel_info_agent: "travel_info_agent",
  accommodation_booking_agent: "accommodation_booking_agent",
} as const;
type AgentTypeKey = keyof typeof AgentType;

const AgentTypeOutputSchema = z.object({
  agent: z.enum(["travel_info_agent", "accommodation_booking_agent"]),
});

const ROUTER_SYSTEM_PROMPT = `You are a router. Given the following user message, decide if it is a travel information question (about destinations, attractions, or general travel info) or an accommodation booking question (about hotels, BnBs, room availability, or prices).
If it is a travel information question, respond with 'travel_info_agent'.
If it is an accommodation booking question, respond with 'accommodation_booking_agent'.`;

export async function buildTravelAssistant() {
  const llm = getLlm();
  const llmRouter = llm.withStructuredOutput(AgentTypeOutputSchema);

  const travelInfoAgent = createReactAgent({
    llm,
    tools: [searchTravelInfoTool, weatherForecastTool],
    stateModifier:
      "You are a helpful assistant that can search travel information and get the weather forecast. " +
      "Only use the tools to find the information you need (including town names).",
  });

  const hotelTools = await getHotelDbToolkitTools();
  const accommodationBookingAgent = createReactAgent({
    llm,
    tools: [...hotelTools, checkBnbAvailabilityTool],
    stateModifier:
      "You are a helpful assistant that can check hotel and BnB room availability and price for a destination in Cornwall. " +
      "You can use the tools to get the information you need. " +
      "If the user does not specify the accommodation type, you should check both hotels and BnBs.",
  });

  async function routerAgentNode(state: typeof MessagesAnnotation.State) {
    const messages = state.messages;
    const lastMsg = messages[messages.length - 1];
    if (lastMsg && lastMsg._getType() === "human") {
      const decision = await llmRouter.invoke([
        new SystemMessage(ROUTER_SYSTEM_PROMPT),
        new HumanMessage(String(lastMsg.content)),
      ]);
      return new Command({ goto: decision.agent as AgentTypeKey });
    }
    return new Command({ goto: AgentType.travel_info_agent });
  }

  const graph = new StateGraph(MessagesAnnotation)
    .addNode("router_agent", routerAgentNode, {
      ends: [AgentType.travel_info_agent, AgentType.accommodation_booking_agent],
    })
    .addNode(AgentType.travel_info_agent, travelInfoAgent)
    .addNode(AgentType.accommodation_booking_agent, accommodationBookingAgent)
    .addEdge("__start__", "router_agent")
    .addEdge(AgentType.travel_info_agent, END)
    .addEdge(AgentType.accommodation_booking_agent, END);

  // MemorySaver = in-memory checkpointer (Python: InMemorySaver). Хранит state по thread_id.
  const checkpointer = new MemorySaver();
  return graph.compile({ checkpointer });
}

// chat_loop из main_08_01.py — обычный диалог с памятью на одном thread_id
export async function chatLoop() {
  const travelAssistant = await buildTravelAssistant();
  const threadId = uuidv1();
  console.log(`Thread ID: ${threadId}`);
  const config = { configurable: { thread_id: threadId } };

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  console.log("UK Travel Assistant (type 'exit' to quit)");
  const ask = () =>
    new Promise<string>((resolve) => rl.question("You: ", (a) => resolve(a.trim())));

  while (true) {
    const userInput = await ask();
    if (["exit", "quit"].includes(userInput.toLowerCase())) break;
    const result = await travelAssistant.invoke(
      { messages: [new HumanMessage(userInput)] },
      config
    );
    const responseMsg = result.messages[result.messages.length - 1];
    console.log(`Assistant: ${responseMsg.content}\n`);
  }
  rl.close();
}

// chat_once_with_rewind из main_08_02.py — демонстрация get_state_history + перемотки.
// Делает один запрос, потом возвращается к последнему checkpoint_id, и задаёт новый
// вопрос со ссылкой на «тот же город» — проверка, что контекст из истории доступен.
export async function chatOnceWithRewind() {
  const travelAssistant = await buildTravelAssistant();
  const threadId = uuidv1();
  console.log(`Thread ID: ${threadId}`);
  const config = { configurable: { thread_id: threadId } };

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  const userInput: string = await new Promise((resolve) =>
    rl.question("You: ", (a) => resolve(a.trim()))
  );
  rl.close();

  const result = await travelAssistant.invoke(
    { messages: [new HumanMessage(userInput)] },
    config
  );
  console.log(`Assistant: ${result.messages[result.messages.length - 1].content}\n`);

  // getStateHistory — стрим всех чекпоинтов. Берём свежий (#J) и его checkpoint_id.
  const snapshots: Awaited<ReturnType<typeof travelAssistant.getState>>[] = [];
  for await (const snap of travelAssistant.getStateHistory(config)) {
    snapshots.push(snap);
  }
  const lastSnapshot = snapshots[0];
  const checkpointId = lastSnapshot.config.configurable?.checkpoint_id as string | undefined;
  console.log(`Last checkpoint_id: ${checkpointId}`);

  const newConfig = {
    configurable: { thread_id: threadId, checkpoint_id: checkpointId },
  };

  // Откатываемся к указанному чекпоинту (#P) — invoke(null, ...) проигрывает граф заново.
  await travelAssistant.invoke(null, newConfig);

  // Новый вопрос на основе контекста из последнего чекпоинта (#Q)
  const result2 = await travelAssistant.invoke(
    { messages: [new HumanMessage("What is the weather in the same town?")] },
    newConfig
  );
  console.log(`Assistant: ${result2.messages[result2.messages.length - 1].content}\n`);
}

if (require.main === module) {
  chatLoop().catch(console.error);
}
