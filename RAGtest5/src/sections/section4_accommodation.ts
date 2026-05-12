/**
 * Секция 4: Агент бронирования (SQL toolkit + BnB).
 * Эквивалент main_04_01.py — отдельный create_react_agent с инструментами SQL-БД отелей
 * (hotel_db_toolkit_tools) и check_bnb_availability. Travel-info агент остаётся как был.
 */
import * as readline from "readline";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { HumanMessage } from "@langchain/core/messages";

import { getLlm } from "../utils/config";
import { searchTravelInfoTool } from "../utils/vectorstore";
import { weatherForecastTool } from "../utils/weather";
import { checkBnbAvailabilityTool } from "../utils/bnb";
import { getHotelDbToolkitTools } from "../utils/hotelDb";

const TOOLS = [searchTravelInfoTool, weatherForecastTool];

export const travelInfoAgent = createReactAgent({
  llm: getLlm(),
  tools: TOOLS,
  stateModifier:
    "You are a helpful assistant that can search travel information and get the weather forecast. " +
    "Only use the tools to find the information you need (including town names).",
});

// Accommodation booking agent (ячейки про accommodation_booking_agent).
// BOOKING_TOOLS = hotel_db_toolkit_tools + [check_bnb_availability]
export async function buildAccommodationAgent() {
  const hotelTools = await getHotelDbToolkitTools();
  const BOOKING_TOOLS = [...hotelTools, checkBnbAvailabilityTool];

  return createReactAgent({
    llm: getLlm(),
    tools: BOOKING_TOOLS,
    stateModifier:
      "You are a helpful assistant that can check hotel and BnB room availability and price for a destination in Cornwall. " +
      "You can use the tools to get the information you need. " +
      "If the user does not specify the accommodation type, you should check both hotels and BnBs.",
  });
}

export async function chatLoop() {
  const accommodationBookingAgent = await buildAccommodationAgent();

  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  console.log("UK Travel Assistant — Accommodation Booking (type 'exit' to quit)");
  const ask = () =>
    new Promise<string>((resolve) => rl.question("You: ", (a) => resolve(a.trim())));

  while (true) {
    const userInput = await ask();
    if (["exit", "quit"].includes(userInput.toLowerCase())) break;
    // В ноутбуке используется accommodation_booking_agent в chat_loop (см. примечание #E)
    const result = await accommodationBookingAgent.invoke({
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
