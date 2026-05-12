/**
 * Секция 1: Базовый LangGraph-агент с ручным ToolsExecutionNode.
 * Эквивалент main_01_01.py / main_01_02.py — единственный tool search_travel_info,
 * собственный класс ToolsExecutionNode (вместо prebuilt ToolNode),
 * условный переход tools_condition между llm_node и tools.
 */
import * as readline from "readline";

/*
 * StateGraph / END / MessagesAnnotation — ядро LangGraph.js.
 * MessagesAnnotation определяет состояние графа с одним полем messages: BaseMessage[],
 * где reducer добавляет новые сообщения к существующему списку (аналог operator.add).
 */
import { StateGraph, END, MessagesAnnotation } from "@langchain/langgraph";

/*
 * ToolNode + toolsCondition — prebuilt узлы LangGraph.
 * toolsCondition возвращает имя следующего узла ("tools" если есть tool_calls, иначе END).
 * Используется как conditional edge между llm_node и tools.
 */
import { toolsCondition } from "@langchain/langgraph/prebuilt";

import { HumanMessage, ToolMessage, AIMessage, BaseMessage } from "@langchain/core/messages";

import { getLlm } from "../utils/config";
import { searchTravelInfoTool } from "../utils/vectorstore";

const TOOLS = [searchTravelInfoTool];

// Биндим tools к LLM (эквивалент llm_model.bind_tools(TOOLS))
const llmWithTools = getLlm().bindTools(TOOLS);

// Ручная реализация ToolsExecutionNode (ячейки #A–#N в ноутбуке)
// Берёт последнее сообщение, проходит по tool_calls, исполняет каждый tool,
// возвращает массив ToolMessage с результатами.
async function toolsExecutionNode(state: typeof MessagesAnnotation.State) {
  const lastMsg = state.messages[state.messages.length - 1] as AIMessage;
  const toolCalls = lastMsg.tool_calls ?? [];

  const toolsByName = Object.fromEntries(TOOLS.map((t) => [t.name, t]));
  const toolMessages: ToolMessage[] = [];

  for (const call of toolCalls) {
    const t = toolsByName[call.name];
    const result = await t.invoke(call.args);
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

// LLM node — простой вызов модели на текущем списке сообщений (ячейки про llm_node)
async function llmNode(state: typeof MessagesAnnotation.State) {
  const response = await llmWithTools.invoke(state.messages as BaseMessage[]);
  return { messages: [response] };
}

// Сборка графа: llm_node ↔ tools, выход через toolsCondition (ячейки #A–#F)
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

// CLI loop (chat_loop из ноутбука)
export async function chatLoop() {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  console.log("UK Travel Assistant (type 'exit' to quit)");

  const ask = () =>
    new Promise<string>((resolve) =>
      rl.question("You: ", (answer) => resolve(answer.trim()))
    );

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
