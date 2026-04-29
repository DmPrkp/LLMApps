# RAGtest1 — мульти-формат загрузка и LCEL-цепочки

Шаг от «голого» Chroma+OpenAI к **LangChain**: разные форматы документов
(PDF / DOCX / TXT / Wikipedia), локальные эмбеддинги HuggingFace и две версии RAG-цепочки —
без памяти и с историей сообщений. Тематика — **Пестум и Чиленто** (юг Италии).

## Стек

- **LangChain.js** — `@langchain/core`, `@langchain/community`, `@langchain/openai`, `@langchain/textsplitters`
- **Локальные эмбеддинги** — `@huggingface/transformers` + `@xenova/transformers`, модель `Xenova/all-MiniLM-L6-v2`
- **ChromaDB** — `chromadb` + интеграция `Chroma` из `@langchain/community/vectorstores/chroma`
- **Загрузчики документов** — `pdf-parse` (PDFLoader), `mammoth` (DocxLoader), `TextLoader`, `WikipediaQueryRun`
- **LLM** — Groq (`llama-3.3-70b-versatile`) через `ChatOpenAI` с кастомным `baseURL`

## Структура

```
src/
├── index.ts                # 5 шагов: clearDb → ingest Paestum → ingest Cilento → search → RAG → RAG with memory
└── utils/
    ├── config.ts           # пути к данным, splitter, embeddings, Chroma instance
    ├── loaders.ts          # clearDb, getLoader, splitAndImport, loadWikipedia, ingestPaestum, ingestFolder
    └── rag.ts              # similaritySearch, buildRagChain, buildRagChainWithMemory
Paestum/                    # 4 файла .docx/.pdf/.txt про Пестум
CilentoTouristInfo/         # 16 файлов .pdf/.docx/.txt про побережье Чиленто
```

## Что делает `index.ts`

1. **`clearDb()`** — стирает все коллекции Chroma на сервере перед запуском.
2. **`ingestPaestum()`** — Wikipedia («Paestum») + 3 локальных файла из `Paestum/`.
3. **`ingestFolder(CILENTO_DIR)`** — пробегает по `CilentoTouristInfo/`, для каждого файла подбирает loader по расширению (`.pdf` / `.docx` / `.txt`) и грузит в Chroma.
4. **Прямой similarity-поиск** по «Where was Poseidonia and who renamed it to Paestum?».
5. **RAG-цепочка без памяти** (`buildRagChain`) — LCEL: `RunnablePassthrough.assign({context: retriever}) → prompt → ChatOpenAI`. Демо: 2 вопроса, второй намеренно проверяет, не выдумает ли модель ответ при отсутствии контекста.
6. **RAG-цепочка с памятью** (`buildRagChainWithMemory`) — добавлен `InMemoryChatMessageHistory` через `placeholder("{chat_history_messages}")` в `ChatPromptTemplate.fromMessages`. После каждого вопроса пишется `HumanMessage` + `AIMessage` в историю. Те же 2 вопроса показывают, что версия с памятью отвечает на анафорическое «А что они потом делают?».

## Ключевые детали

- **Чанкинг** — `RecursiveCharacterTextSplitter({ chunkSize: 500, chunkOverlap: 0 })`
- **Метаданные** — нормализуются в `splitAndImport` (null/object → строка), чтобы Chroma не падал
- **Имя коллекции** — `tourist_info_hf4` (`hf` = HuggingFace embeddings)
- В `splitAndImport` фильтруются пустые чанки (`pageContent.trim().length > 0`)

## Запуск

```bash
npm install
echo 'GROQ_API_KEY=...' > .env

# нужен запущенный ChromaDB локально
npm start
# или для отладки PDF-парсинга:
ts-node src/debug-pdf.ts
```

## Главная идея

Показать **полный конвейер RAG на LangChain** с тремя важными штуками сверх RAGInit:
мульти-форматные loaders, LCEL-композиция через `pipe`, и chat-история как
дополнительный input цепочки.
