/**
 * Секция 7: MCP-клиент (стаб).
 * Эквивалент main_07_01.py — добавляем к travel_info_agent инструменты от MCP-сервера
 * AccuWeather, который крутится отдельным Python-процессом (mcp/accuweather_mcp.py).
 *
 * В отличие от Python-версии этот файл — СТАБ:
 *   • В TS не подключаем зависимость @langchain/mcp-adapters, потому что для боевого
 *     запуска нужен живой FastMCP-сервер (Python) — он не входит в скоуп переноса.
 *   • Здесь показан паттерн интеграции: где взять MCP-tools и как смешать их с локальным
 *     search_travel_info перед передачей в create_react_agent.
 *
 * Чтобы получить рабочую версию:
 *   1) запусти Python MCP-сервер: `python building-llm-applications/ch11/mcp/accuweather_mcp.py`
 *   2) добавь зависимость:        `npm i @langchain/mcp-adapters`
 *   3) раскомментируй блок REAL ниже и удали стабовый weatherForecastTool.
 */
import * as readline from "readline";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { HumanMessage } from "@langchain/core/messages";

import { getLlm } from "../utils/config";
import { searchTravelInfoTool } from "../utils/vectorstore";
import { weatherForecastTool } from "../utils/weather";

// ----- REAL (закомментирован) -----
// import { MultiServerMCPClient } from "@langchain/mcp-adapters";
//
// async function getAccuweatherTools() {
//   const client = new MultiServerMCPClient({
//     accuweather: {
//       url: "http://127.0.0.1:8020/accu-mcp-server",
//       transport: "streamable_http",
//     },
//   });
//   return await client.getTools();
// }

// ----- STUB -----
// Возвращаем мок weather tool вместо живых tools от MCP-сервера AccuWeather.
async function getAccuweatherTools() {
  console.warn(
    "[section7] MCP-сервер не подключён. Используется мок weather_forecast вместо AccuWeather. " +
      "Подробности — в шапке файла."
  );
  return [weatherForecastTool];
}

export async function buildTravelInfoAgent() {
  const accuweatherTools = await getAccuweatherTools();
  const tools = [searchTravelInfoTool, ...accuweatherTools];

  return createReactAgent({
    llm: getLlm(),
    tools,
    name: "travel_info_agent",
    stateModifier:
      "You are a helpful assistant that can search travel information and get the weather forecast. " +
      "Only use the tools to find the information you need (including town names).",
  });
}

export async function chatLoop() {
  const agent = await buildTravelInfoAgent();
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  console.log("UK Travel Assistant (type 'exit' to quit)");
  const ask = () =>
    new Promise<string>((resolve) => rl.question("You: ", (a) => resolve(a.trim())));

  while (true) {
    const userInput = await ask();
    if (["exit", "quit"].includes(userInput.toLowerCase())) break;
    const result = await agent.invoke({ messages: [new HumanMessage(userInput)] });
    const responseMsg = result.messages[result.messages.length - 1];
    console.log(`Assistant: ${responseMsg.content}\n`);
  }
  rl.close();
}

if (require.main === module) {
  chatLoop().catch(console.error);
}
