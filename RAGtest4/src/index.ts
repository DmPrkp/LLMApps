import { runDataIngestion } from "./sections/section1_ingest";
import { runMetadataFiltering } from "./sections/section2_metadata";
import { runSqlQueryGeneration } from "./sections/section3_sql";
import { runQueryRouter } from "./sections/section4_router";
import { runRagFusion } from "./sections/section5_rag_fusion";

async function main() {
  // Раскомментировать для повторной загрузки данных (только при первом запуске или сбросе):
  // await runDataIngestion();

  // await runMetadataFiltering();
  await runSqlQueryGeneration();
  // await runQueryRouter();
  // await runRagFusion();
}

main().catch(console.error);
