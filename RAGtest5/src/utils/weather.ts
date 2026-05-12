/**
 * WeatherForecastService (мок) + weather_forecast tool.
 * Эквивалент классов и tool-а из main_02_01.py / main_02_02.py.
 */
import { tool, type StructuredToolInterface } from "@langchain/core/tools";
import { z } from "zod";

export type Weather = "sunny" | "foggy" | "rainy" | "windy";

export interface WeatherForecast {
  town: string;
  weather: Weather;
  temperature: number;
}

const WEATHER_OPTIONS: Weather[] = ["sunny", "foggy", "rainy", "windy"];
const TEMP_MIN = 18;
const TEMP_MAX = 31;

// Метод get_forecast() из WeatherForecastService (мок REST-клиента к AccuWeather)
export function getMockForecast(town: string): WeatherForecast {
  const weather = WEATHER_OPTIONS[Math.floor(Math.random() * WEATHER_OPTIONS.length)];
  const temperature = TEMP_MIN + Math.floor(Math.random() * (TEMP_MAX - TEMP_MIN + 1));
  return { town, weather, temperature };
}

export const weatherForecastTool: StructuredToolInterface = tool(
  async ({ town }: { town: string }) => {
    const forecast = getMockForecast(town);
    if (!forecast) {
      return { error: `No weather data available for '${town}'.` };
    }
    return forecast;
  },
  {
    name: "weather_forecast",
    description: "Get the weather forecast, given a town name.",
    schema: z.object({
      town: z.string().describe("The name of the town to get the weather forecast for."),
    }),
  }
);
