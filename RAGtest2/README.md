# RAGtest2 — стратегии чанкинга и продвинутые ретриверы

Семь стратегий разбиения и индексации документов, чтобы сравнить **что улучшает retrieval**:
гранулярное vs крупное разбиение, родительские документы, мульти-векторные ретриверы
с резюме, гипотетическими вопросами и расширенным («оконным») контекстом.
Тематика — **Корнуолл и Восточный Сассекс** (Wikivoyage).

## Стек

- **LangChain.js** — `langchain/retrievers/parent_document`, `langchain/retrievers/multi_vector`, `langchain/storage/in_memory`
- **ChromaDB** — `Chroma` из `@langchain/community`
- **Локальные эмбеддинги** — `Xenova/all-MiniLM-L6-v2` через `HuggingFaceTransformersEmbeddings`
- **HTML-парсинг** — `cheerio` + `html-to-text`
- **LLM** — Groq `llama-3.3-70b-versatile` (нужен только для секций ⑤ и ⑥)
- **`zod`** — структурированный вывод гипотетических вопросов через `withStructuredOutput`
- **`uuid`** — ключи для связывания дочерних чанков с родительскими

## Структура

```
src/
├── index.ts                # все 7 стратегий как функции; в main() раскомментируется одна
└── utils/
    ├── config.ts           # 4 splitter-а, embeddings, список UK-направлений, GROQ_API_KEY
    └── htmlUtils.ts        # loadHtmlDocument, htmlToTextDocs, splitByHtmlSections, resetChromaCollection
```

> В `package.json` есть скрипты `section1`–`section6`, указывающие на файлы в
> `src/sections/*` — этих файлов нет, всё реализовано в `index.ts` как функции.
> Запускайте `npm start` и комментируйте/раскомментируйте нужный вызов в `main()`.

## Семь стратегий

### ① `runGranularVsCoarseSingleUrl` — гранулярное vs крупное (один URL)
Один URL Корнуолла, две коллекции: `cornwall_granular` (разбиение по H2-секциям через
`splitByHtmlSections`) и `cornwall_coarse` (HTML→текст + `RecursiveCharacterTextSplitter`,
chunkSize=3000, overlap=300). Сравнение topK=1 на «Hotels in Cornwall».

### ② `runMultiUrlGranularVsCoarse` — то же на множестве URL
Те же две стратегии, но прогон по всему `ukDestinationUrls` (в `config.ts` сейчас
включён только Cornwall — раскомментируйте остальные направления). Запросы:
«Hotels in East Sussex» и «Beaches in Cornwall».

### ③ `runParentDocumentRetriever` — родительские документы
`ParentDocumentRetriever`: индексируем мелкие дочерние чанки в Chroma, но возвращаем
**крупные родительские** из `InMemoryStore`. Запрос «Cornwall Ranger».

### ④ `runMultiVectorChildChunks` — мульти-векторный с дочерними чанками
То же, но через `MultiVectorRetriever` + `byteStore` — больше контроля над связкой
«мелкое в индексе ↔ крупное в storage» через `metadata.doc_id` (uuid).

### ⑤ `runMultiVectorSummaries` — мульти-векторный с LLM-резюме
В Chroma идут **резюме** крупных чанков (через цепочку `prompt → chatbot → StringOutputParser`),
а в storage — оригинальные крупные чанки. Поиск по плотным резюме, ответ — на полном тексте.
Медленнее всех (LLM-вызов на каждый чанк).

### ⑥ `runMultiVectorHypotheticalQuestions` — мульти-векторный с гипотетическими вопросами
Для каждого крупного чанка LLM генерирует 4 гипотетических вопроса (через
`withStructuredOutput(zod-схема)`), они индексируются. Идея: пользовательский вопрос
семантически ближе к гипотетическим вопросам, чем к самому тексту. Запрос: «Как добраться из Лондона в Корнуолл?».

### ⑦ `runMultiVectorExpandedContext` — оконный контекст
В Chroma идут гранулярные чанки, в storage — «расширенные» = `[предыдущий, текущий, следующий]`
склеенные. Поиск точечный, ответ — на расширенном окне.

## Запуск

```bash
npm install
echo 'GROQ_API_KEY=...' > .env   # нужен только для ⑤ и ⑥

# нужен запущенный ChromaDB локально

npm start                         # запустит то, что не закомментировано в main()
```

В `main()` сейчас активна только `runMultiVectorExpandedContext()` — переключайте на нужную.

## Главная идея

Один **корпус** (Wikivoyage UK) обрабатывается семью способами, чтобы вживую увидеть
trade-off **«размер чанка ↔ полнота контекста ↔ точность retrieval»**: маленькие чанки
лучше попадают в запрос, но теряют контекст; родительские/мульти-векторные стратегии
возвращают полный контекст; индексация по резюме/вопросам сдвигает семантику ближе к
пользовательскому языку.
