interface Vacancy {
  name: string;
  salary?: Salary;
  employer?: {
    name: string;
  };
  snippet?: {
    requirement?: string;
    responsibility?: string;
  };
  alternate_url: string;
  category?: string;
}

interface FetchResult {
  items: Vacancy[];
  limitReached: boolean;
}

interface Salary {
  from?: number;
  to?: number;
  currency: string;
}

interface CompactVacancy {
  title: string;
  company: string;
  salary_from: number | null;
  salary_to: number | null;
  currency: string;
  description: string;
  skills: string[];
  id: string;
}

export {
  Salary,
  FetchResult,
  Vacancy,
  CompactVacancy
}
