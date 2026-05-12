/**
 * Секция 5: Роутер агентов через structured output.
 * Эквивалент main_05_01.py — узел router_agent_node по системному промпту и
 * structured-output классификатору решает, какому из двух агентов (travel_info_agent
 * или accommodation_booking_agent) передать запрос, и возвращает Command(goto=...).
 */
import * as readline from "readline";
import { z } from "zod";

import { StateGraph, END, MessagesAnnotation, Command } from "@langchain/langgraph";
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

// Zod-схема structured output (эквивалент Pydantic AgentTypeOutput)
const AgentTypeOutputSchema = z.object({
  agent: z
    .enum(["travel_info_agent", "accommodation_booking_agent"])
    .describe("Which agent should handle the query?"),
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

  // router_agent_node (ячейки #A–#I в main_05_01.py)
  async function routerAgentNode(state: typeof MessagesAnnotation.State) {
    const messages = state.messages;
    const lastMsg = messages[messages.length - 1];

    if (lastMsg && lastMsg._getType() === "human") {
      const userInput = String(lastMsg.content);
      const decision = await llmRouter.invoke([
        new SystemMessage(ROUTER_SYSTEM_PROMPT),
        new HumanMessage(userInput),
      ]);
      return new Command({ goto: decision.agent as AgentTypeKey });
    }
    // По умолчанию идём в travel_info_agent (#I)
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

  return graph.compile();
}

export async function chatLoop() {
  const travelAssistant = await buildTravelAssistant();
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  console.log("UK Travel Assistant (type 'exit' to quit)");
  const ask = () =>
    new Promise<string>((resolve) => rl.question("You: ", (a) => resolve(a.trim())));

  while (true) {
    const userInput = await ask();
    if (["exit", "quit"].includes(userInput.toLowerCase())) break;
    const result = await travelAssistant.invoke({
      messages: [new HumanMessage(userInput)],
    });
    const responseMsg = result.messages[result.messages.length - 1];
    console.log(`Assistant: ${responseMsg.content}\n`);
  }
  rl.close();
}

if (require.main === module) {
  chatLoop().catch(console.error);
}
