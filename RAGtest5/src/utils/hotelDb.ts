/**
 * SQLDatabaseToolkit-аналог для отеля Cornwall_hotels.
 * Поднимает SQLite в памяти через sql.js (как в RAGtest4) и отдаёт tools, эквивалентные
 * SQLDatabaseToolkit из langchain_community.agent_toolkits:
 *   sql_db_list_tables    — список таблиц
 *   sql_db_schema         — CREATE-схема для указанных таблиц
 *   sql_db_query_checker  — заглушка-валидатор SQL
 *   sql_db_query          — выполнить SELECT
 */
import * as fs from "fs";
import * as path from "path";
import initSqlJs, { Database } from "sql.js";
import { tool, type StructuredToolInterface } from "@langchain/core/tools";
import { z } from "zod";

const HOTEL_DB_DIR = path.join(__dirname, "../../hotel_db");

let _sql: Awaited<ReturnType<typeof initSqlJs>> | null = null;
let _db: Database | null = null;

async function getSqlJs() {
  if (!_sql) {
    const wasmPath = path.resolve(require.resolve("sql.js"), "../../dist/sql-wasm.wasm");
    _sql = await initSqlJs({
      wasmBinary: fs.readFileSync(wasmPath) as unknown as ArrayBuffer,
    });
  }
  return _sql;
}

// Ленивая инициализация одной общей БД в памяти. Поднимаем схему + сиды.
export async function getHotelDb(): Promise<Database> {
  if (_db) return _db;
  const SQL = await getSqlJs();
  _db = new SQL.Database();
  const schemaSql = fs.readFileSync(path.join(HOTEL_DB_DIR, "cornwall_hotels_schema.sql"), "utf8");
  _db.run(schemaSql);
  return _db;
}

function listTables(db: Database): string[] {
  const res = db.exec("SELECT name FROM sqlite_master WHERE type='table'");
  if (!res.length) return [];
  return res[0].values.map((row) => String(row[0]));
}

function tableSchemas(db: Database, names: string[]): string {
  if (names.length === 0) return "";
  const placeholders = names.map(() => "?").join(",");
  const stmt = db.prepare(
    `SELECT sql FROM sqlite_master WHERE type='table' AND name IN (${placeholders})`
  );
  const parts: string[] = [];
  stmt.bind(names);
  while (stmt.step()) {
    const row = stmt.get();
    if (row[0]) parts.push(String(row[0]) + ";");
  }
  stmt.free();
  return parts.join("\n\n");
}

function runQuery(db: Database, sql: string): Record<string, unknown>[] {
  const res = db.exec(sql);
  if (!res.length) return [];
  const { columns, values } = res[0];
  return values.map((row) =>
    Object.fromEntries(columns.map((c, i) => [c, row[i]]))
  );
}

// Срезаем markdown-блоки и преамбулу из вывода LLM (повторяет cleanSql из RAGtest4).
function cleanSql(raw: string): string {
  const cleaned = raw
    .replace(/```sql\n?/gi, "")
    .replace(/```\n?/gi, "")
    .replace(/^(here is|here's|the sql|sql query|query).*?:\s*/gi, "")
    .trim();
  const match = cleaned.match(/SELECT[\s\S]+?;?$/i);
  return match ? match[0].trim() : cleaned;
}

export async function getHotelDbToolkitTools(): Promise<StructuredToolInterface[]> {
  const db = await getHotelDb();

  const listTablesTool: StructuredToolInterface = tool(
    async () => listTables(db).join(", "),
    {
      name: "sql_db_list_tables",
      description: "List all tables available in the hotel database.",
      schema: z.object({}),
    }
  );

  const schemaTool: StructuredToolInterface = tool(
    async ({ table_names }: { table_names: string }) => {
      const names = table_names
        .split(",")
        .map((n) => n.trim())
        .filter(Boolean);
      return tableSchemas(db, names);
    },
    {
      name: "sql_db_schema",
      description:
        "Get the CREATE TABLE schema and sample rows for a comma-separated list of tables. " +
        "Call sql_db_list_tables first to see available tables.",
      schema: z.object({
        table_names: z
          .string()
          .describe("Comma-separated table names, e.g. 'hotels, hotel_room_offers'."),
      }),
    }
  );

  const queryCheckerTool: StructuredToolInterface = tool(
    async ({ query }: { query: string }) => {
      const lower = query.toLowerCase();
      if (!lower.includes("select")) {
        return "Query must be a SELECT statement.";
      }
      return query;
    },
    {
      name: "sql_db_query_checker",
      description:
        "Validate a SQLite SELECT query before running it. Returns the (possibly fixed) query.",
      schema: z.object({
        query: z.string().describe("The SQLite SELECT query to validate."),
      }),
    }
  );

  const queryTool: StructuredToolInterface = tool(
    async ({ query }: { query: string }) => {
      const cleaned = cleanSql(query);
      try {
        const rows = runQuery(db, cleaned);
        return JSON.stringify(rows);
      } catch (err) {
        return `SQL error: ${(err as Error).message}`;
      }
    },
    {
      name: "sql_db_query",
      description:
        "Execute a SQLite SELECT query against the hotel database and return the rows as JSON.",
      schema: z.object({
        query: z.string().describe("A valid SQLite SELECT query."),
      }),
    }
  );

  return [listTablesTool, schemaTool, queryCheckerTool, queryTool];
}
