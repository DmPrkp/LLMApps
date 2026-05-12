import { chatLoop as section1 } from "./sections/section1_basic_agent";
import { chatLoop as section2 } from "./sections/section2_multi_tool";
import { chatLoop as section3 } from "./sections/section3_react_agent";
import { chatLoop as section4 } from "./sections/section4_accommodation";
import { chatLoop as section5 } from "./sections/section5_router";
import { chatLoop as section6 } from "./sections/section6_supervisor";
import { chatLoop as section7 } from "./sections/section7_mcp";
import { chatLoop as section8 } from "./sections/section8_checkpointer";
import { chatLoop as section9 } from "./sections/section9_guardrails";

async function main() {
  // Раскомментируй одну из секций. Каждая запускает интерактивный CLI чат.
  // await section1();  // ① Базовый LangGraph с ToolsExecutionNode
  // await section2();  // ② + weather_forecast + system message
  // await section3();  // ③ Prebuilt createReactAgent
  // await section4();  // ④ Accommodation booking agent (SQL + BnB)
  // await section5();  // ⑤ Роутер через structured output
  // await section6();  // ⑥ Supervisor (multi-agent)
  // await section7();  // ⑦ MCP (стаб)
  // await section8();  // ⑧ Checkpointer (память)
  await section9();     // ⑨ Guardrails (refusal классификатор)
}

main().catch(console.error);
