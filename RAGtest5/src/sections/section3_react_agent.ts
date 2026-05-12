/**
 * Секция 3: Prebuilt React-агент.
 * Эквивалент main_03_01.py — заменяем самописный граф на createReactAgent из @langchain/langgraph/prebuilt.
 * Внутри он сам делает llm_node + tool_node + tools_condition.
 */
import * as readline from "readline";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { HumanMessage } from "@langchain/core/messages";

import { getLlm } from "../utils/config";
import { searchTravelInfoTool } from "../utils/vectorstore";
import { weatherForecastTool } from "../utils/weather";

const TOOLS = [searchTravelInfoTool, weatherForecastTool];

// createReactAgent скрывает state schema, ToolsExecutionNode и conditional edges.
// Prompt задаётся одним полем (как stateModifier / messagesModifier в LangGraph.js).
export const travelInfoAgent = createReactAgent({
  llm: getLlm(),
  tools: TOOLS,
  stateModifier:
    "You are a helpful assistant that can search travel information and get the weather forecast. " +
    "Only use the tools to find the information you need (including town names).",
});

export async function chatLoop() {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  console.log("UK Travel Assistant (type 'exit' to quit)");
  const ask = () =>
    new Promise<string>((resolve) => rl.question("You: ", (a) => resolve(a.trim())));

  while (true) {
    const userInput = await ask();
    if (["exit", "quit"].includes(userInput.toLowerCase())) break;
    const result = await travelInfoAgent.invoke({
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
