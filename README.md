# llmAppTest — серия RAG-экспериментов на TypeScript

Перенос примеров из книги **«Building LLM Applications»** (главы 8–10) с
Python/Jupyter на TypeScript/Node.js. Каждая папка — отдельный проект на ту же тему,
но с нарастающей сложностью.

Общий стек по серии: **LangChain.js**, **ChromaDB**, локальные эмбеддинги
**HuggingFace** (`Xenova/all-MiniLM-L6-v2` и `Xenova/bge-base-en-v1.5`), LLM **Groq
`llama-3.3-70b-versatile`** через OpenAI-совместимый endpoint.

## Проекты

| Папка | Тема | Что внутри |
|---|---|---|
| [RAGInit](./RAGInit/README.md) | Самый простой RAG | Голый ChromaDB + OpenAI SDK без LangChain. Naive vs safer prompt на провокационных вопросах про Пестум. |
| [RAGtest1](./RAGtest1/README.md) | Мульти-формат + LCEL + память | LangChain loaders (PDF/DOCX/TXT/Wikipedia), LCEL-цепочки, `InMemoryChatMessageHistory`. Корпус — Пестум и Чиленто. |
| [RAGtest2](./RAGtest2/README.md) | Стратегии чанкинга и ретриверы | 7 стратегий: гранулярное vs крупное разбиение, `ParentDocumentRetriever`, `MultiVectorRetriever` с резюме / гипотетическими вопросами / расширенным контекстом. Корпус — Wikivoyage UK. |
| [RAGtest3](./RAGtest3/README.md) | Трансформация запросов | Rewrite-Retrieve-Read, Multi-Query, Step-back questions, HyDE. Тот же корпус UK Wikivoyage. |
| [RAGtest4](./RAGtest4/README.md) | Метаданные, SQL, роутинг, RAG Fusion | Фильтрация по метаданным через Zod, NL→SQL поверх SQLite (`sql.js`), роутер «Chroma vs SQL», RAG Fusion с Reciprocal Rank Fusion. |

## Логика прогрессии

```
RAGInit       →   RAGtest1      →   RAGtest2        →   RAGtest3        →   RAGtest4
─────────         ─────────         ──────────          ──────────          ──────────
без LangChain     LangChain         улучшаем            улучшаем            гибридное
3 шага RAG        + loaders         индекс              запрос              хранилище
                  + память                                                  (Chroma + SQL)
```

## Общие требования

- **Node.js** + **TypeScript** (`ts-node` для запуска, `tsc` для билда).
- **ChromaDB** локально (например, `docker run -p 8000:8000 chromadb/chroma`) — нужен всем проектам кроме голого RAGInit (там тоже нужен).
- **`GROQ_API_KEY`** в `.env` каждой папки — единственный внешний секрет.
- **Эмбеддинги** работают локально на CPU через ONNX Runtime — внешние API для них не нужны.
- Для **RAGtest4** дополнительно нужны SQL-файлы из `building-llm-applications/ch10/` (`CreateUkBooking.sql`, `PopulateUkBooking.sql`).

## Прочее в репозитории

- `building-llm-applications/` — оригинальные ноутбуки книги (Python).
- `chroma-data/` — локальные данные ChromaDB.
- `hh-parser/` — отдельный проект, не относится к серии RAG.
- `model.txt` — заметки.
