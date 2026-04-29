# RAGtest4 — продвинутые техники RAG на TypeScript

Перенос главы 10 книги «Building LLM Applications» с Python/Jupyter на TypeScript/Node.js.
Пять секций демонстрируют пайплайн RAG поверх туристических данных Wikivoyage по UK
(Корнуолл и Восточный Сассекс) и SQLite-базы бронирований.

## Стек

- **LangChain.js** (`@langchain/core`, `@langchain/community`, `@langchain/openai`) — цепочки, промпты, парсеры, retriever-ы
- **ChromaDB** (`chromadb` + `@langchain/community/vectorstores/chroma`) — векторное хранилище
- **HuggingFace Transformers** (`@huggingface/transformers`) — локальные эмбеддинги `Xenova/bge-base-en-v1.5` (без внешних API)
- **Groq API** через OpenAI-совместимый endpoint — LLM `llama-3.3-70b-versatile`
- **sql.js** — SQLite в памяти через WASM
- **cheerio** + **html-to-text** — парсинг HTML-страниц
- **zod** — схемы для структурированного вывода LLM

## Структура

```
src/
├── index.ts                       # точка входа, раскомментируй нужную секцию
├── sections/
│   ├── section1_ingest.ts         # ① загрузка Wikivoyage → Chroma с метаданными
│   ├── section2_metadata.ts       # ② фильтрация по метаданным + структурированный запрос
│   ├── section3_sql.ts            # ③ NL → SQL (text-to-SQL)
│   ├── section4_router.ts         # ④ роутер: Chroma vs SQL
│   └── section5_rag_fusion.ts     # ⑤ RAG Fusion с Reciprocal Rank Fusion
└── utils/
    ├── config.ts                  # эмбеддинги, ключи, маппинг направление→регион
    ├── dbUtils.ts                 # инициализация SQLite, выполнение запросов, очистка SQL
    └── htmlUtils.ts               # загрузка HTML, разбиение по h2, сброс Chroma
sql/
├── CreateUkBooking.sql            # схема таблиц UK Booking
└── PopulateUkBooking.sql          # тестовые данные (отели, предложения)
```

## Что делает каждая секция

### ① `section1_ingest.ts` — загрузка данных с метаданными
Берёт URL'ы из `UK_DESTINATION_REGIONS` (21 направление по Корнуоллу и Восточному Сассексу),
скачивает HTML с Wikivoyage, режет на чанки по заголовкам H2, к каждому чанку добавляет
метаданные `destination` + `region` и складывает в коллекцию Chroma `uk_metadata`.

> Запускается один раз перед остальными секциями (раскомментировать в `index.ts`).

### ② `section2_metadata.ts` — фильтрация по метаданным
Две техники:
- **А) ручная фильтрация** — `similaritySearch` с явным фильтром `{ destination: "Newquay" }`
- **Б) структурированный запрос** — LLM по Zod-схеме `DestinationSearch` извлекает
  `content_search` / `destination` / `region` из вопроса, из них собирается фильтр Chroma
  с `$and` / `$eq`, и только потом выполняется поиск + RAG-ответ.

### ③ `section3_sql.ts` — text-to-SQL
Поднимает SQLite в памяти из `sql/CreateUkBooking.sql` + `sql/PopulateUkBooking.sql`,
отдаёт LLM схему таблиц, генерирует SQL по вопросу («Give me some offers for Cardiff…»),
очищает вывод (`cleanSql` срезает markdown и преамбулу) и выполняет через `sql.js`.

### ④ `section4_router.ts` — роутер источников
LLM решает, куда отправить вопрос:
- `tourist_info_store` → векторный поиск в Chroma (советы, достопримечательности)
- `uk_booking_db` → text-to-SQL по базе бронирований (предложения, цены)

Затем результаты любого пути передаются в общий RAG-промпт. Тестируется на двух
вопросах: про предложения в Брайтоне (→ SQL) и про пляжи Корнуолла (→ Chroma).

### ⑤ `section5_rag_fusion.ts` — RAG Fusion
1. LLM генерирует 5 переформулировок исходного вопроса.
2. По каждой делается retrieval (4 документа).
3. Алгоритм **Reciprocal Rank Fusion** (`score = Σ 1/(rank + k)`, k=60) переранжирует
   все найденные документы.
4. Топ-3 идёт в контекст финального RAG-ответа.

## Запуск

```bash
npm install

# .env — единственная переменная
echo 'GROQ_API_KEY=...' > .env

# Chroma должна быть запущена локально на дефолтном порту
# (например: docker run -p 8000:8000 chromadb/chroma)

# первый запуск — наполняем коллекцию
# (раскомментировать `runDataIngestion()` в src/index.ts)
npm start

# отдельные секции
npm run ingest      # ① загрузка
npm run metadata    # ② метаданные
npm run sql         # ③ SQL
npm run router      # ④ роутер
npm run fusion      # ⑤ RAG fusion
```

## Внешние зависимости

- Запущенный сервер ChromaDB на `localhost`.
- `GROQ_API_KEY` в `.env`.
