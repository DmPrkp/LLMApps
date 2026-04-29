# RAGInit — самый простой RAG

Точка старта серии: «голый» RAG из 2-3 этапов без LangChain. Векторное хранилище —
ChromaDB с дефолтным эмбеддером, LLM — Groq через OpenAI SDK. Тексты — три абзаца про
**Пестум** (древнегреческий город на юге Италии) из Britannica.

## Стек

- `chromadb` + `chromadb-default-embed` — векторное хранилище и встроенные эмбеддинги
- `openai` SDK — но с `baseURL` Groq (`https://api.groq.com/openai/v1`), модель `llama-3.3-70b-versatile`
- `dotenv` — `GROQ_API_KEY` из `.env`
- `@anthropic-ai/sdk` — заявлен в зависимостях, но в коде не используется

## Структура

```
src/
├── index.ts   # пайплайн ingest → query → 3 демо-вопроса
└── utils.ts   # createCollection / ingestDocuments / queryVectorDatabase / executeLlmPrompt / myChatbot
```

## Что делает `index.ts`

1. Создаёт коллекцию `tourism_collection` и добавляет 3 жёстко зашитых документа про Пестум.
2. **Demo:** прямой запрос к векторной БД («Сколько дорических храмов в Пестуме?»).
3. **Trick question / naive prompt:** «Сколько колонн у трёх храмов вместе взятых?» — запрос без guardrails (модель будет фантазировать).
4. **Trick question / safer prompt** через `myChatbot`: системный промпт «отвечай только из контекста, иначе — Я не знаю».
5. **Полноценный вопрос:** «Сколько храмов в Пестуме, кто их построил и в каком стиле?» через `myChatbot`.

`myChatbot` — каноничный 3-шаговый RAG: `queryVectorDatabase` → `promptTemplate` → `executeLlmPrompt`.

## Запуск

```bash
npm install
echo 'GROQ_API_KEY=...' > .env

# нужен запущенный ChromaDB на localhost:8000
npm run dev          # ts-node src/index.ts
# или: npm run build && npm start
```

## Главная идея

Сравнить «наивный» промпт (LLM врёт на провокационных вопросах) и **безопасный**
(`prompt template` с инструкцией «не знаешь — скажи Я не знаю»). Это база, на которой
строятся остальные RAGtest*.
