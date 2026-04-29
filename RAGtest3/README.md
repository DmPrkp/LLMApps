# RAGtest3 — трансформация запросов

Если RAGtest2 улучшает **индекс**, то RAGtest3 улучшает **запрос**: переписывание,
multi-query, step-back и HyDE. Корпус тот же — UK-направления Wikivoyage,
гранулярная (по H2) Chroma-коллекция `uk_granular`.

## Стек

- **LangChain.js** — `@langchain/core` (промпты, парсеры, runnable), `@langchain/community` (Chroma), `@langchain/openai` (Groq)
- **ChromaDB** — Chroma в режиме клиента
- **Локальные эмбеддинги** — `Xenova/all-MiniLM-L6-v2`
- **HTML-парсинг** — `cheerio` + `html-to-text`
- **LLM** — Groq `llama-3.3-70b-versatile`
- **`zod`** — схема для multi-query JSON-вывода
- **`langchain/retrievers/multi_query`** — `MultiQueryRetriever`

## Структура

```
src/
├── index.ts                       # все 5 секций как функции — переключаются в main()
├── sections/                      # дубликаты тех же секций как самостоятельные скрипты
│   ├── section1_ingest.ts
│   ├── section2_rewriter.ts
│   ├── section3_multi_query.ts
│   ├── section4_step_back.ts
│   └── section5_hyde.ts
└── utils/
    ├── config.ts                  # 21 направление UK, embeddings, GROQ_API_KEY
    └── htmlUtils.ts               # loadHtmlDocument, splitDocsIntoGranularChunks, resetChromaCollection
```

## Пять секций

### ① `runDataIngestion` — загрузка данных
Сбрасывает коллекцию `uk_granular`, грузит HTML по всем URL из `ukDestinationUrls`,
режет по H2-секциям и пишет в Chroma. Запускается один раз перед остальными секциями.

### ② `runQueryRewriter` — Rewrite-Retrieve-Read
Сравнивает прямой поиск с поиском после **переписывания запроса** LLM:
«Tell me some fun things I can enjoy in Cornwall» → «leisure activities in Cornwall».
Затем стандартный RAG-промпт даёт финальный ответ.

### ③ `runMultiQuery` — Multi-Query Generation
LLM по zod-схеме `{questions: string[]}` генерирует 5 вариаций исходного вопроса,
а потом то же самое делается через готовый `MultiQueryRetriever.fromLLM`. Цель —
обойти ограничения dense-retrieval за счёт нескольких формулировок.

### ④ `runStepBack` — Step-back Questions
LLM с few-shot-примерами «отступает» к более **общему** вопросу
(«tips for a trip to Brighton» → «what makes Brighton a popular tourist destination»),
затем по нему делается retrieval, а ответ — на исходный вопрос. Лучше работает с узкими
запросами, где буквальный поиск даёт мало контекста.

### ⑤ `runHyDE` — Hypothetical Document Embeddings
LLM сначала пишет **гипотетическое предложение-ответ** на вопрос, потом эмбеддинг этого
предложения используется для поиска (а не эмбеддинг вопроса). Идея: гипотетический ответ
ближе к стилю документов, чем сам вопрос. Запрос: «What are the best beaches in Cornwall?».

## Запуск

```bash
npm install
echo 'GROQ_API_KEY=...' > .env

# нужен запущенный ChromaDB локально

# первый запуск — наполняем коллекцию
# (раскомментировать `runDataIngestion()` в src/index.ts → main)
npm start

# отдельные секции через скрипты:
npm run ingest      # ① загрузка
npm run rewriter    # ② переписывание
npm run multiquery  # ③ multi-query
npm run stepback    # ④ step-back
npm run hyde        # ⑤ HyDE
```

## Главная идея

«Не модифицируй индекс — модифицируй запрос». Все четыре техники (②–⑤) работают
поверх **одной и той же** Chroma-коллекции и показывают, как разная подготовка
запроса меняет полноту найденного контекста.
