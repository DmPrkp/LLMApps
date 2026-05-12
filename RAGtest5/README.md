# RAGtest5 — мульти-агент travel-ассистент на LangGraph.js

Девять секций последовательно собирают UK travel-ассистента: от простого
LangGraph с одним tool-ом до supervisor-а двух агентов с guardrails и памятью.

## Стек

- **LangGraph.js** (`@langchain/langgraph`) — графы агентов: StateGraph, MessagesAnnotation, ToolNode, createReactAgent, Command, MemorySaver
- **LangGraph Supervisor** (`@langchain/langgraph-supervisor`) — multi-agent supervisor
- **LangChain.js** (`@langchain/core`, `@langchain/community`, `@langchain/openai`) — tool-ы, сообщения, structured output, MemoryVectorStore
- **HuggingFace Transformers** (`@huggingface/transformers`) — локальные эмбеддинги `Xenova/bge-base-en-v1.5`
- **Groq API** через OpenAI-совместимый endpoint — LLM `llama-3.3-70b-versatile`
- **sql.js** — SQLite в памяти (WASM)
- **html-to-text** — конвертация Wikivoyage HTML в plain text
- **zod** — схемы для bind_tools и withStructuredOutput

## Структура

```
src/
├── index.ts                          # точка входа — раскомментируй нужную секцию
├── sections/
│   ├── section1_basic_agent.ts       # ① Custom StateGraph с ToolsExecutionNode (main_01_01/02.py)
│   ├── section2_multi_tool.ts        # ② + weather_forecast + SystemMessage (main_02_01/02.py)
│   ├── section3_react_agent.ts       # ③ Prebuilt createReactAgent (main_03_01.py)
│   ├── section4_accommodation.ts     # ④ Accommodation agent: SQL toolkit + BnB (main_04_01.py)
│   ├── section5_router.ts            # ⑤ Routing через withStructuredOutput + Command (main_05_01.py)
│   ├── section6_supervisor.ts        # ⑥ createSupervisor multi-agent (main_06_01.py)
│   ├── section7_mcp.ts               # ⑦ MCP-клиент (стаб) (main_07_01.py)
│   ├── section8_checkpointer.ts      # ⑧ MemorySaver + state history + rewind (main_08_01/02.py)
│   └── section9_guardrails.ts        # ⑨ Router guardrail + pre-model guardrail (main_09_01/02.py)
└── utils/
    ├── config.ts                     # env vars, фабрика LLM, embeddings, UK_DESTINATIONS
    ├── vectorstore.ts                # build_vectorstore + searchTravelInfoTool (MemoryVectorStore)
    ├── weather.ts                    # WeatherForecastService mock + weatherForecastTool
    ├── bnb.ts                        # BnBBookingService mock + checkBnbAvailabilityTool
    └── hotelDb.ts                    # sql.js-based SQLDatabaseToolkit-аналог (4 tool-а)
hotel_db/
└── cornwall_hotels_schema.sql        # схема + сиды для SQLite-БД отелей Корнуолла
```

## Что делает каждая секция

### ① `section1_basic_agent.ts` — базовый LangGraph

- Граф из двух узлов: `llm_node` ↔ `tools` (custom `ToolsExecutionNode`)
- Conditional edge через `toolsCondition` (если есть tool_calls → `tools`, иначе → END)
- Единственный tool: `search_travel_info` (RAG поверх Wikivoyage UK)

### ② `section2_multi_tool.ts` — два tool-а + system prompt

- Добавляем `weather_forecast` (мок)
- `SystemMessage` инжектится в каждом вызове `llmNode`

### ③ `section3_react_agent.ts` — prebuilt React

- Заменяем самописный граф на `createReactAgent` из `@langchain/langgraph/prebuilt`
- Тот же поведенческий контракт, меньше кода

### ④ `section4_accommodation.ts` — Accommodation booking agent

- SQLite-БД отелей через `sql.js` → 4 tool-а (`sql_db_list_tables` / `_schema` / `_query_checker` / `_query`) — аналог `SQLDatabaseToolkit`
- `check_bnb_availability` поверх мок-списка BnB
- Отдельный ReAct-агент с этими 5 tool-ами

### ⑤ `section5_router.ts` — структурный роутер

- `withStructuredOutput(zod)` определяет, в какой агент пойти
- Граф: `router_agent` → (`travel_info_agent` | `accommodation_booking_agent`) → END
- Каждый узел возвращает `Command({ goto: ... })`

### ⑥ `section6_supervisor.ts` — supervisor

- Меняем самописный router на `createSupervisor` из `@langchain/langgraph-supervisor`
- Supervisor может звать обоих агентов и координировать ответы
- Каждый sub-agent имеет `name` (важно для supervisor tool_calls)

### ⑦ `section7_mcp.ts` — MCP-клиент (СТАБ)

- В Python-версии подключаются tools от FastMCP-сервера AccuWeather (`http://127.0.0.1:8020/accu-mcp-server`)
- В этой TS-версии — стаб с моком, потому что MCP-сервер (Python) выходит за рамки переноса
- В шапке файла — инструкция, как включить реальный MCP через `@langchain/mcp-adapters`

### ⑧ `section8_checkpointer.ts` — память графа

- `MemorySaver` хранит state по `thread_id`
- `chatLoop()` ведёт диалог в одном треде — следующее сообщение видит историю
- `chatOnceWithRewind()` показывает `getStateHistory` + откат к предыдущему `checkpoint_id`

### ⑨ `section9_guardrails.ts` — guardrails

- **Уровень 1 (роутер):** `llmGuardrail` решает, travel-related ли вопрос. Если нет — `Command({ update: { messages: [AIMessage(refusal)] }, goto: "guardrail_refusal" })` → END.
- **Уровень 2 (агент):** `stateModifier` каждого ReAct-агента работает как `pre_model_hook`: перед каждым вызовом LLM повторно классифицирует и при необходимости инжектит `SystemMessage` с инструкцией отказа.

## Запуск

```bash
npm install

# .env

# В src/index.ts раскомментировать нужную секцию.
npm start

# Или напрямую любую секцию:
npm run section1   # ① базовый граф
npm run section2   # ② multi-tool
npm run section3   # ③ React agent
npm run section4   # ④ accommodation
npm run section5   # ⑤ router
npm run section6   # ⑥ supervisor
npm run section7   # ⑦ MCP (stub)
npm run section8   # ⑧ checkpointer
npm run section9   # ⑨ guardrails
```

## Отличия от Python ch11

| Что                 | Python ch11                                          | RAGtest5 (TS)                                               |
| ------------------- | ---------------------------------------------------- | ----------------------------------------------------------- |
| LLM                 | `gpt-5`, `gpt-5-mini` (OpenAI Responses API)         | `llama-3.3-70b-versatile` (Groq)                            |
| Эмбеддинги          | `OpenAIEmbeddings`                                   | локальный `Xenova/bge-base-en-v1.5`                         |
| Векторное хранилище | `Chroma` (требует сервер)                            | `MemoryVectorStore` (в памяти)                              |
| SQL-БД              | `SQLDatabase` на файле + `SQLDatabaseToolkit`        | `sql.js` (WASM) + самописные 4 tool-а                       |
| MCP                 | Живой `MultiServerMCPClient`                         | Стаб с моком (см. шапку `section7_mcp.ts`)                  |
| `pre_model_hook`    | Параметр `pre_model_hook` у `create_react_agent`     | `stateModifier` как async-функция                           |
| Loader Wikivoyage   | `AsyncHtmlLoader` + `RecursiveCharacterTextSplitter` | `fetch` + `html-to-text` + `RecursiveCharacterTextSplitter` |

## Внешние зависимости

- **Только `GROQ_API_KEY`** в `.env`.
- Chroma НЕ нужна (используем `MemoryVectorStore`).
- При первом запуске любой секции скачиваются 4 страницы Wikivoyage и считаются эмбеддинги (~30–60 сек). После этого `getTravelInfoRetriever()` отдаёт кэш.
