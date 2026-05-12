/**
 * Секция 9: Guardrails — фильтруем не-travel вопросы.
 * Эквивалент main_09_01.py / main_09_02.py:
 *   1) (09_01) В router_agent_node перед routing вызываем llm_guardrail (structured output).
 *      Если is_travel=false — кладём в state AIMessage с отказом и идём в guardrail_refusal → END.
 *   2) (09_02) Дополнительно прокидываем pre_model_hook в каждый ReAct-агент: если хук
 *      решит что вопрос не-travel, инжектит SystemMessage с REFUSAL_INSTRUCTION перед моделью.
 *      В LangGraph.js это аналог stateModifier как функции, оборачивающей сообщения.
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
import {
  HumanMessage,
  SystemMessage,
  AIMessage,
  BaseMessage,
} from "@langchain/core/messages";

import { getLlm } from "../utils/config";
import { searchTravelInfoTool } from "../utils/vectorstore";
import { weatherForecastTool } from "../utils/weather";
import { checkBnbAvailabilityTool } from "../utils/bnb";
import { getHotelDbToolkitTools } from "../utils/hotelDb";

// ----- Routing -----
const AgentType = {
  travel_info_agent: "travel_info_agent",
  accommodation_booking_agent: "accommodation_booking_agent",
  guardrail_refusal: "guardrail_refusal",
} as const;
type AgentTypeKey = keyof typeof AgentType;

const AgentTypeOutputSchema = z.object({
  agent: z.enum(["travel_info_agent", "accommodation_booking_agent"]),
});

const ROUTER_SYSTEM_PROMPT = `You are a router. Given the following user message, decide if it is a travel information question (about destinations, attractions, or general travel info) or an accommodation booking question (about hotels, BnBs, room availability, or prices).
If it is a travel information question, respond with 'travel_info_agent'.
If it is an accommodation booking question, respond with 'accommodation_booking_agent'.`;

// ----- Guardrail -----
const GuardrailDecisionSchema = z.object({
  is_travel: z
    .boolean()
    .describe(
      "True if the user question is about travel information: destinations, attractions, lodging (hotels/BnBs), prices, availability, or weather in Cornwall/England."
    ),
  reason: z.string().describe("Brief justification for the decision."),
});

const GUARDRAIL_SYSTEM_PROMPT =
  "You are a strict classifier. Given the user's last message, respond with whether it is " +
  "travel-related. Travel-related queries include destinations, attractions, lodging (hotels/BnBs), " +
  "room availability, prices, or weather in Cornwall/England.";

const AGENT_GUARDRAIL_SYSTEM_PROMPT =
  "You are a strict classifier. Given the user's last message, respond with whether it is travel-related. " +
  "Travel-related queries include destinations, attractions, lodging (hotels/BnBs), room availability, " +
  "prices, or weather in Cornwall/England. Only accept travel-related questions covering Cornwall " +
  "(England) and reject any questions from other areas in England and from other countries.";

const AGENT_REFUSAL_INSTRUCTION =
  "You can only help with travel-related questions (destinations, attractions, lodging, prices, " +
  "availability, or weather in Cornwall/England). The user's request is not travel-related. " +
  "Or it might be a travel related question but not focusing on Cornwall (England). " +
  "Politely refuse and briefly explain what topics you can help with.";

const ROUTER_REFUSAL_TEXT =
  "Sorry, I can only help with travel-related questions (destinations, attractions, " +
  "lodging, prices, availability, or weather in Cornwall/England). Please rephrase your " +
  "request to be travel-related.";

export async function buildTravelAssistant() {
  const llm = getLlm();
  const llmRouter = llm.withStructuredOutput(AgentTypeOutputSchema);
  const llmGuardrail = llm.withStructuredOutput(GuardrailDecisionSchema);

  // pre_model_hook эквивалент: stateModifier-функция, оборачивающая сообщения
  // перед каждым вызовом LLM внутри ReAct-агента. Сигнатура: (state, config) => BaseMessage[].
  // Если последнее human-message не-travel — инжектим SystemMessage с REFUSAL_INSTRUCTION.
  // (Ячейки про pre_model_guardrail в main_09_02.py.)
  async function preModelGuardrail(
    state: { messages: BaseMessage[] }
  ): Promise<BaseMessage[]> {
    const messages = state.messages;
    const last = messages[messages.length - 1];
    if (!last || last._getType() !== "human") return [...messages];
    const decision = await llmGuardrail.invoke([
      new SystemMessage(AGENT_GUARDRAIL_SYSTEM_PROMPT),
      new HumanMessage(String(last.content)),
    ]);
    if (decision.is_travel) return [...messages];
    return [new SystemMessage(AGENT_REFUSAL_INSTRUCTION), ...messages];
  }

  const travelInfoAgent = createReactAgent({
    llm,
    tools: [searchTravelInfoTool, weatherForecastTool],
    stateModifier: preModelGuardrail,
  });

  const hotelTools = await getHotelDbToolkitTools();
  const accommodationBookingAgent = createReactAgent({
    llm,
    tools: [...hotelTools, checkBnbAvailabilityTool],
    stateModifier: preModelGuardrail,
  });

  // router_agent_node + первый уровень guardrail (ячейки 09_01)
  async function routerAgentNode(state: typeof MessagesAnnotation.State) {
    const messages = state.messages;
    const lastMsg = messages[messages.length - 1];
    if (lastMsg && lastMsg._getType() === "human") {
      const userInput = String(lastMsg.content);

      const decision = await llmGuardrail.invoke([
        new SystemMessage(GUARDRAIL_SYSTEM_PROMPT),
        new HumanMessage(userInput),
      ]);

      if (!decision.is_travel) {
        // Кладём AIMessage с отказом и шорткатим граф через guardrail_refusal
        return new Command({
          update: { messages: [new AIMessage(ROUTER_REFUSAL_TEXT)] },
          goto: AgentType.guardrail_refusal,
        });
      }

      const routed = await llmRouter.invoke([
        new SystemMessage(ROUTER_SYSTEM_PROMPT),
        new HumanMessage(userInput),
      ]);
      return new Command({ goto: routed.agent as AgentTypeKey });
    }
    return new Command({ goto: AgentType.travel_info_agent });
  }

  // No-op узел, только для того чтобы выйти в END с уже подменённым AIMessage в state
  function guardrailRefusalNode(_state: typeof MessagesAnnotation.State) {
    return {};
  }

  const graph = new StateGraph(MessagesAnnotation)
    .addNode("router_agent", routerAgentNode, {
      ends: [
        AgentType.travel_info_agent,
        AgentType.accommodation_booking_agent,
        AgentType.guardrail_refusal,
      ],
    })
    .addNode(AgentType.travel_info_agent, travelInfoAgent)
    .addNode(AgentType.accommodation_booking_agent, accommodationBookingAgent)
    .addNode(AgentType.guardrail_refusal, guardrailRefusalNode)
    .addEdge("__start__", "router_agent")
    .addEdge(AgentType.travel_info_agent, END)
    .addEdge(AgentType.accommodation_booking_agent, END)
    .addEdge(AgentType.guardrail_refusal, END);

  const checkpointer = new MemorySaver();
  return graph.compile({ checkpointer });
}

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

if (require.main === module) {
  chatLoop().catch(console.error);
}
