/**
 * Секция 2: Два tool-а + системный промпт.
 * Эквивалент main_02_01.py / main_02_02.py — добавляем weather_forecast tool,
 * прокидываем SystemMessage в llm_node чтобы задать поведение ассистента.
 * Граф такой же как в секции 1 (custom ToolsExecutionNode).
 */
import * as readline from "readline";

import { StateGraph, END, MessagesAnnotation } from "@langchain/langgraph";
import { toolsCondition } from "@langchain/langgraph/prebuilt";
import {
  HumanMessage,
  SystemMessage,
  ToolMessage,
  AIMessage,
  BaseMessage,
} from "@langchain/core/messages";

import { getLlm } from "../utils/config";
import { searchTravelInfoTool } from "../utils/vectorstore";
import { weatherForecastTool } from "../utils/weather";

const TOOLS = [searchTravelInfoTool, weatherForecastTool];
const llmWithTools = getLlm().bindTools(TOOLS);

// System prompt из main_02_02.py (ячейки про llm_node)
const SYSTEM_PROMPT = `You are a helpful assistant that can search travel information and get the weather forecast.
Only use the tools to find the information you need (including town names).`;

async function toolsExecutionNode(state: typeof MessagesAnnotation.State) {
  const lastMsg = state.messages[state.messages.length - 1] as AIMessage;
  const toolCalls = lastMsg.tool_calls ?? [];
  const toolsByName = Object.fromEntries(TOOLS.map((t) => [t.name, t]));
  const toolMessages: ToolMessage[] = [];

  for (const call of toolCalls) {
    const result = await toolsByName[call.name].invoke(call.args);
    toolMessages.push(
      new ToolMessage({
        content: JSON.stringify(result),
        name: call.name,
        tool_call_id: call.id!,
      })
    );
  }
  return { messages: toolMessages };
}

async function llmNode(state: typeof MessagesAnnotation.State) {
  // System message добавляется к каждому вызову (main_02_02.py)
  const messagesWithSystem: BaseMessage[] = [
    new SystemMessage(SYSTEM_PROMPT),
    ...(state.messages as BaseMessage[]),
  ];
  const response = await llmWithTools.invoke(messagesWithSystem);
  return { messages: [response] };
}

const builder = new StateGraph(MessagesAnnotation)
  .addNode("llm_node", llmNode)
  .addNode("tools", toolsExecutionNode)
  .addEdge("__start__", "llm_node")
  .addConditionalEdges("llm_node", toolsCondition, {
    tools: "tools",
    [END]: END,
  })
  .addEdge("tools", "llm_node");

export const travelInfoAgent = builder.compile();

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
