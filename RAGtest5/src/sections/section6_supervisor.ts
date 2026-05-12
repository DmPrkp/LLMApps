/**
 * Секция 6: Supervisor multi-agent pattern.
 * Эквивалент main_06_01.py — заменяем самодельный routerAgentNode на готовый createSupervisor
 * из @langchain/langgraph-supervisor. Supervisor сам решает, какому(им) агенту(ам) делегировать
 * запрос и комбинирует их ответы.
 */
import * as readline from "readline";

/*
 * createSupervisor — фабрика supervisor-графа из @langchain/langgraph-supervisor.
 * Принимает массив подагентов и системный промпт. Внутри строит граф, где supervisor
 * вызывает agents tool-like через tool_calls, а сами агенты возвращают результат в supervisor.
 */
import { createSupervisor } from "@langchain/langgraph-supervisor";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { HumanMessage } from "@langchain/core/messages";

import { getLlm } from "../utils/config";
import { searchTravelInfoTool } from "../utils/vectorstore";
import { weatherForecastTool } from "../utils/weather";
import { checkBnbAvailabilityTool } from "../utils/bnb";
import { getHotelDbToolkitTools } from "../utils/hotelDb";

const SUPERVISOR_PROMPT = `You are a supervisor that manages two agents: a travel information agent and an accommodation booking agent.
You can answer user questions that might require calling both agents when needed.
Decide which agent(s) to use for each user request and coordinate their responses.`;

export async function buildTravelAssistant() {
  const llm = getLlm();

  // Имя агента важно: supervisor использует его как tool name (ячейка про name=...)
  const travelInfoAgent = createReactAgent({
    llm,
    tools: [searchTravelInfoTool, weatherForecastTool],
    name: "travel_info_agent",
    stateModifier:
      "You are a helpful assistant that can search travel information and get the weather forecast. " +
      "Only use the tools to find the information you need (including town names).",
  });

  const hotelTools = await getHotelDbToolkitTools();
  const accommodationBookingAgent = createReactAgent({
    llm,
    tools: [...hotelTools, checkBnbAvailabilityTool],
    name: "accommodation_booking_agent",
    stateModifier:
      "You are a helpful assistant that can check hotel and BnB room availability and price for a destination in Cornwall. " +
      "You can use the tools to get the information you need. " +
      "If the user does not specify the accommodation type, you should check both hotels and BnBs.",
  });

  const supervisor = createSupervisor({
    agents: [travelInfoAgent, accommodationBookingAgent],
    llm: getLlm(),
    supervisorName: "travel_assistant",
    prompt: SUPERVISOR_PROMPT,
  });

  return supervisor.compile();
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
