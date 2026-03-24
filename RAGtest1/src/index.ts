import * as dotenv from "dotenv";
dotenv.config();

import { CILENTO_DIR } from "./utils/config";
import { clearDb, ingestPaestum, ingestFolder } from "./utils/loaders";
import { similaritySearch, buildRagChain, executeChain, buildRagChainWithMemory } from "./utils/rag";

async function main(): Promise<void> {
  await clearDb();

  console.log(' 1. Ingest Paestum documents')
  await ingestPaestum();

  console.log('2. Ingest CilentoTouristInfo folder')
  await ingestFolder(CILENTO_DIR);

  console.log('3. Direct similarity search')
  await similaritySearch("Where was Poseidonia and who renamed it to Paestum?");

  console.log("4. RAG chain (no memory)");
  const ragChain = await buildRagChain();

  const q1 = "Где находилась Посейдония, и кто переименовал её в Пестум? Также укажите источник информации.";
  console.log(`\nQ: ${q1}`);
  console.log("A:", await executeChain(ragChain, q1));

  const q2 = "А что они потом делают? Скажите, только если знаете. Также сообщите источник.";
  console.log(`\nQ: ${q2}`);
  console.log("A:", await executeChain(ragChain, q2));

  // 5. RAG chain with memory
  console.log("\n=== RAG chain with memory ===");
  const executeChainWithMemory = await buildRagChainWithMemory();

  const q3 = "Где находилась Посейдония, и кто переименовал её в Пестум? Также укажите источник информации.";
  console.log(`\nQ: ${q3}`);
  console.log("A:", await executeChainWithMemory(q3));

  const q4 = "А что они потом делают? Скажите, только если знаете. Также сообщите источник.";
  console.log(`\nQ: ${q4}`);
  console.log("A:", await executeChainWithMemory(q4));
}

main().catch(console.error);
