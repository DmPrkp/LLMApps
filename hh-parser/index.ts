import Groq from 'groq-sdk';
import * as dotenv from 'dotenv';
import { CompactVacancy, Salary, Vacancy } from './types.js';
import { CATEGORIES, categorizeVacancy, fetchVacanciesPage } from './hh-fetch.js';
import fs from 'fs'

// Пороговые значения зарплат
const SALARY_THRESHOLD_RUB = 300000;
const SALARY_THRESHOLD_USD = 3000;
const PAGES = 10;
const CHUNK_SIZE = 20;

dotenv.config();

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY
});

const MODEL = process.env.GROQ_MODEL || "meta-llama/llama-4-scout-17b-16e-instruct"

async function runMinimalPrompt(prompt: string, content: string): Promise<string> {
  try {
    const chatCompletion = await groq.chat.completions.create({
      messages: [{
        role: "system",
        content
      }, { role: "user", content: prompt }],
      model: MODEL,
      temperature: 0,
      max_tokens: 35000
    });

    return chatCompletion.choices[0]?.message?.content || '';
  } catch (error: any) {
    console.error("❌ Ошибка в запросе:", error.message);
    return '';
  }
}

function extractSkills(text: string): string[] {
  const commonSkills = [
    'Python', 'Java', 'C++', 'C#', 'JavaScript', 'TypeScript', 'Go', 'Rust',
    'React', 'Node.js', 'Django', 'Flask', 'Spring', 'SQL', 'NoSQL',
    'AWS', 'Docker', 'Kubernetes', 'Git', 'CI/CD', 'REST API',
    'Machine Learning', 'Data Science', 'AI', 'TensorFlow', 'PyTorch'
  ];

  const found: string[] = [];
  const lowerText = text.toLowerCase();

  for (const skill of commonSkills) {
    if (lowerText.includes(skill.toLowerCase())) {
      found.push(skill);
    }
  }

  return found.slice(0, 5); // Оставляем максимум 5 навыков
}

function prepareVacanciesForLLM(vacancies: any[]): CompactVacancy[] {
  return vacancies.map(vac => {
    // Извлекаем ключевые навыки из требований
    const requirement = vac.snippet?.requirement || '';
    // const skills = extractSkills(requirement);

    return {
      title: vac.name || 'Без названия',
      company: vac.employer?.name || 'Не указана',
      salary_from: vac.salary?.from || null,
      salary_to: vac.salary?.to || null,
      currency: vac.salary?.currency || 'RUR',
      description: (vac.snippet?.requirement || '') + ' ' + (vac.snippet?.responsibility || ''),
      skills: requirement,
      id: vac.id
    };
  });
}


function isSalaryAboveThreshold(salary?: Salary): boolean {
  if (!salary) return false;

  const checkValue = (value?: number): boolean => {
    if (!value) return false;
    if (salary.currency === 'RUR' && value >= SALARY_THRESHOLD_RUB) return true;
    if (salary.currency === 'USD' && value >= SALARY_THRESHOLD_USD) return true;
    if (salary.currency === 'EUR' && value * 1.1 >= SALARY_THRESHOLD_USD) return true;
    return false;
  };

  return checkValue(salary.from);
}

export async function collectHighSalaryVacancies(): Promise<Vacancy[]> {
  let highSalaryVacancies: Vacancy[] = [];
  let page = 0;
  let limitReached = false;
  let shouldStop = false;
  let cnt = 0;

  console.log(
    "🔍 Начинаем сбор IT-вакансий с сортировкой по убыванию зарплаты..."
  );
  console.log(
    `💰 Порог: ${SALARY_THRESHOLD_RUB}₽ или ${SALARY_THRESHOLD_USD}$`
  );
  console.log(`📋 Фильтр: только программисты и IT-специалисты\n`);

  while (page < PAGES && !limitReached && !shouldStop) {
    console.log(`📄 Загрузка страницы ${page + 1}...`);

    const { items, limitReached: reached } = await fetchVacanciesPage(
      "developer OR программист OR разработчик OR engineer",
      "1",
      page
    );

    limitReached = reached;

    if (items.length === 0) {
      console.log("  ⚠️ Страница пустая, останавливаемся");
      break;
    }

    const pageHighSalary = items.filter(v => isSalaryAboveThreshold(v.salary));

    const pageWithCategories = pageHighSalary.map((v) => {
      const category = categorizeVacancy(v);
      return { ...v, category };
    })

    const pageLowSalary = items.filter(v => !isSalaryAboveThreshold(v.salary));

    highSalaryVacancies = highSalaryVacancies.concat(pageWithCategories);

    console.log(
      `  ✅ Высоких: ${pageHighSalary.length}, ❌ Низких: ${pageLowSalary.length}`
    );

    if (pageWithCategories.length > 0) {
      const examples = pageWithCategories.slice(0, 3).map(v => {
        const titleMatch = v.category && CATEGORIES[v.category]?.some(k =>
          v.name?.toLowerCase().includes(k)
        );
        return `${v.category}${titleMatch ? ' (в названии)' : ' (в описании)'}`;
      }).join(', ');
      console.log(`  🔍 Найдено: ${examples}`);
    }

    if (pageHighSalary.length === 0 && highSalaryVacancies.length > 0) {
      ++cnt;
    }

    if (
      pageHighSalary.length === 0 &&
      highSalaryVacancies.length > 0 &&
      cnt > 1
    ) {
      console.log(
        `\n🛑 Останавливаемся: на странице ${page + 1} нет вакансий выше порога`
      );
      shouldStop = true;
      break;
    }

    page++;

    if (!limitReached && !shouldStop) {
      await new Promise((r) => setTimeout(r, 500));
    }
  }

  console.log(`\n✅ Собрано вакансий: ${highSalaryVacancies.length}`);

  return highSalaryVacancies;
}



function chunkArray<T>(array: T[], chunkSize: number): T[][] {
  const chunks = [];
  for (let i = 0; i < array.length; i += chunkSize) {
    chunks.push(array.slice(i, i + chunkSize));
  }
  return chunks;
}

function parseModelResponse(response: string): any[] {
  // Убираем markdown-обёртки
  let cleanResponse = response.replace(/```(?:json)?\s*|\s*```/g, '').trim();

  cleanResponse = cleanResponse.replace(/<think>[\s\S]*?<\/think>/g, '').trim();

  const firstBracket = cleanResponse.indexOf('[');
  const lastBracket = cleanResponse.lastIndexOf(']');
  if (firstBracket !== -1 && lastBracket !== -1 && lastBracket > firstBracket) {
    cleanResponse = cleanResponse.substring(firstBracket, lastBracket + 1);
  }

  try {
    const parsed = JSON.parse(cleanResponse);
    if (Array.isArray(parsed)) {
      console.log(`✅ Распарсен массив из ${parsed.length} элементов`);
      return parsed;
    } else {
      console.log("⚠️ Распарсено, но это не массив:", typeof parsed);
      return [];
    }
  } catch (e) {
    console.log("❌ Ошибка парсинга даже после очистки:", cleanResponse.substring(0, 100));
    return [];
  }
}

async function main() {
  const allVacancies = await collectHighSalaryVacancies();
  console.log(`✅ Всего вакансий: ${allVacancies.length}`);

  const chunks = chunkArray(prepareVacanciesForLLM(allVacancies), CHUNK_SIZE);

  // Собираем "сырые" результаты
  const candidates: any[] = [];

  for (let i = 0; i < chunks.length; i++) {
    console.log(`\n🔄 Часть ${i + 1}/${chunks.length}`);

    const prompt = `Из этого списка вакансий выбери ТОЛЬКО те, которые подходят под критерии:
      - Программисты/разработчики (не менеджеры, не тестировщики, не devops, не фронтендеры)
      - Высокая зарплата (относительно других в списке)
          
      Верни ТОЛЬКО массив с объектами выбранных вакансий:
    ${JSON.stringify(chunks[i], null, 2)}`;

    const content = `Ты - анализатор вакансий. Твоя задача - отбирать только вакансии чистых программистов/разработчиков и devops.
          ПРАВИЛА ОТБОРА:
          ✅ ОСТАВЛЯЕМ: backend, fullstack, C++, C#, Python, Java, Go, Rust, разработчик, software engineer, devops, 1С разработчик
          ❌ ИСКЛЮЧАЕМ: frontend, QA, тестировщик, менеджер, director, CTO, lead, администратор

          ВАЖНО: Твой ответ должен быть ТОЛЬКО валидным JSON-массивом.
          Не используй никакие пояснения, теги (<think>, \`\`\` и т.д.), вводные фразы.
          Ответ должен начинаться с символа '[' и заканчиваться ']'.
          Также максимально сократить описание вакансии и оставить 
          - коротко описание например геймдев или бэкенд 
          - специализацию, 
          - языки 
          - зарплату
          - заголовок
          Никакого дополнительного текста до или после массива.`

    const result = await runMinimalPrompt(prompt, content);

    candidates.push(parseModelResponse(result))

    fs.writeFileSync(
      "vacancies.json",
      JSON.stringify(candidates, null, 2),
    );

    if (i < chunks.length - 1) {
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }

  // Финальный запрос для составления топа
  console.log("\n🤔 Составляю итоговый топ...");

  const finalPrompt = `Проанализируй все эти вакансии программистов:

    ${JSON.stringify(candidates, null, 2)}

    Формат ответа: сгруппируй по специализациям и языкам, например:

    - Backend на Node.js (включая NestJS) – 2 шт 
    - .NET‑backend на C# – 2 шт
    - Backend на Rust / C++ – 1 шт  
    - Data Engineering на Python / SQL – 1 шт  
    - C++‑разработка на C++ – 1 шт
    - Embedded на C / C++ / Linux‑kernel – 1 шт  
    - Backend FinTech на PHP / Symfony / Python – 1 шт
    - Backend (не указано) – 1 шт
      и тд..

    не достаточно просто написать бэкенд нужно указать язык или языки
    не достаточно просто написать разработка нужно из описания понять какая сфера геймдев, embedded или бэкенд`;

  const content2 = 'Ты советчик и аналитик отвечаешь цифрами и точными данными (не JSON) но человекочитемыми.'

  const finalResult = await runMinimalPrompt(finalPrompt, content2);

  console.log("\n" + "=".repeat(50));
  console.log("🏆 ТОП-10 ВЫСОКООПЛАЧИВАЕМЫХ ВАКАНСИЙ:");
  console.log("=".repeat(50));
  console.log(finalResult);
}

main()