import { FetchResult, Vacancy } from "./types.js";

// Коды профессиональных ролей для IT-специалистов (из справочника hh.ru)
const IT_ROLES = [
  "96", // Программист
  "112", // DevOps
  "121", // Архитектор
  "124", // Web-разработчик
  "125", // Мобильный разработчик
  "126", // Embedded разработчик
  "127", // Game Developer
  "128", // 1С разработчик
  "129", // SQL разработчик
];

// Категории и ключевые слова для поиска в названии вакансии
export const CATEGORIES: Record<string, string[]> = {
  "C++": ["c++", "cpp", "c/c++", "с++", "embedded", "системный программист"],
  "C#": ["c#", "csharp", ".net", "asp.net"],
  ML: [
    "machine learning",
    "ml",
    "deep learning",
    "ai",
    "нейросети",
    "искусственный интеллект",
  ],
  Python: ["python"],
  Java: ["java"],
  Go: ["go", "golang"],
  Rust: ["rust"],
  Node: ["node", "node.js", "nodejs"],
  PHP: ["php"],
  Ruby: ["ruby"],
  Swift: ["swift"],
  Kotlin: ["kotlin"],
  "1C": ["1c", "1с", "1с:предприятие"],
  DevOps: ["devops"],
  Frontend: [
    "frontend",
    "front-end",
    "front end",
    "vue",
    "angular",
    "react",
    "react.js",
    "reactjs",
    "redux",
  ],
  Flutter: ["flutter"],
  SQL: ["sql", "postgresql", "mysql", "базы данных", "database"],
  Scala: ["scala"],
  Elixir: ["elixir"],
  Clojure: ["clojure"],
  Haskell: ["haskell"],
};


export async function fetchVacanciesPage(
  text: string,
  area: string = "113",
  page: number = 0
): Promise<FetchResult> {
  const url = new URL("https://api.hh.ru/vacancies");
  url.searchParams.set("text", text);
  url.searchParams.set("area", area);
  url.searchParams.set("only_with_salary", "true");
  url.searchParams.set("per_page", "100");
  url.searchParams.set("page", page.toString());
  url.searchParams.set("order_by", "salary_desc");

  IT_ROLES.forEach((role) => {
    url.searchParams.append("professional_role", role);
  });

  const response = await fetch(url, {
    headers: {
      "User-Agent": `SalaryCollector/1.0 (${process.env.RU_EMAIL})`,
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    const errorText = await response.text();
    if (response.status === 400 && errorText.includes("more than 2000 items")) {
      return { items: [], limitReached: true };
    }
    throw new Error(`HTTP ${response.status}: ${errorText}`);
  }

  const data = await response.json();

  return {
    items: data.items,
    limitReached: false,
  };
}

export function categorizeVacancy(vacancy: Vacancy): string {
  const title = vacancy.name?.toLowerCase() || "";

  let description = "";
  if (vacancy.snippet) {
    const requirement = vacancy.snippet.requirement || "";
    const responsibility = vacancy.snippet.responsibility || "";
    description = cleanHtml(requirement + " " + responsibility);
  }

  for (const [category, keywords] of Object.entries(CATEGORIES)) {
    for (const keyword of keywords) {
      if (title.includes(keyword.toLowerCase())) {
        return category;
      }
    }
  }

  for (const [category, keywords] of Object.entries(CATEGORIES)) {
    for (const keyword of keywords) {
      if (description.includes(keyword.toLowerCase())) {
        return category;
      }
    }
  }

  if (
    title.includes("программист") ||
    title.includes("developer") ||
    title.includes("разработчик")
  ) {
    return "Other";
  }

  if (
    title.includes("team lead") ||
    title.includes("тимлид") ||
    title.includes("ведущий")
  ) {
    return "Team Lead";
  }

  if (title.includes("архитектор") || title.includes("architect")) {
    return "Architect";
  }

  if (
    title.includes("director") ||
    title.includes("директор") ||
    title.includes("cto")
  ) {
    return "Director/CTO";
  }

  return "Other";
}

function cleanHtml(htmlText?: string): string {
  if (!htmlText) return "";
  return htmlText
    .replace(/<[^>]*>/g, " ")
    .replace(/&[a-z]+;/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}