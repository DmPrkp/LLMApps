import * as fs from "fs";
import * as path from "path";
import initSqlJs, { Database } from "sql.js";

// Путь к SQL-файлам внутри проекта
const SQL_DIR = path.join(__dirname, "../../sql");

// Кэшируем экземпляр sql.js чтобы не загружать WASM дважды
let _sql: Awaited<ReturnType<typeof initSqlJs>> | null = null;

async function getSqlJs() {
  if (!_sql) {
    // Указываем путь к бандлу WASM из пакета sql.js
    const wasmPath = path.resolve(
      require.resolve("sql.js"),
      "../../dist/sql-wasm.wasm"
    );
    _sql = await initSqlJs({
      wasmBinary: fs.readFileSync(wasmPath) as unknown as ArrayBuffer,
    });
  }
  return _sql;
}

// Создаём базу данных в памяти и заполняем из SQL-файлов
export async function getDatabase(): Promise<Database> {
  const SQL = await getSqlJs();
  const db = new SQL.Database();

  const createSql = fs.readFileSync(
    path.join(SQL_DIR, "CreateUkBooking.sql"),
    "utf8"
  );
  const populateSql = fs.readFileSync(
    path.join(SQL_DIR, "PopulateUkBooking.sql"),
    "utf8"
  );

  db.run(createSql);
  db.run(populateSql);
  return db;
}

// Получаем CREATE-схемы всех таблиц для передачи в промпт LLM
export function getTableSchemas(db: Database): string {
  const results = db.exec(
    "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
  );
  if (!results.length) return "";
  return results[0].values.map((row) => row[0] + ";").join("\n\n");
}

// Выполняем SELECT и возвращаем результат как массив объектов
export function execQuery(
  db: Database,
  sql: string
): Record<string, unknown>[] {
  const results = db.exec(sql);
  if (!results.length) return [];
  const { columns, values } = results[0];
  return values.map((row) =>
    Object.fromEntries(columns.map((col, i) => [col, row[i]]))
  );
}

// Убираем markdown-блоки и преамбулу из вывода LLM, оставляем чистый SQL
export function cleanSql(rawSql: string): string {
  const cleaned = rawSql
    .replace(/```sql\n?/gi, "")
    .replace(/```\n?/gi, "")
    .replace(/^(here is|here's|the sql|sql query|query).*?:\s*/gi, "")
    .trim();
  const sqlMatch = cleaned.match(/SELECT[\s\S]+?;/i);
  return sqlMatch ? sqlMatch[0] : cleaned;
}
