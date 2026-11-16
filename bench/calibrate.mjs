#!/usr/bin/env node
/**
 * calibrate.mjs - Auto-calibración Few-Shot de Temperatura y Sesgo para TokenSafe Remaster (#48)
 */

import { loadItems, systemOneRemaster } from "./eval_accuracy.mjs";

async function main() {
    const limit = parseInt(process.argv[2] || "5", 10);
    console.log(`=== Auto-Calibración Few-Shot (${limit} muestras por tarea) ===\n`);
    const items = loadItems(null, limit);

    const byTask = new Map();
    for (const it of items) {
        const { answers, diagnostics } = await systemOneRemaster(it.state, it.questions, true);
        if (!byTask.has(it.task)) byTask.set(it.task, []);
        byTask.get(it.task).push({ item: it, ans: answers.q, diag: diagnostics.q });
    }

    for (const [task, records] of byTask.entries()) {
        const kind = records[0].item.kind;
        let ok = 0;
        for (const r of records) {
            if (kind === "choice" && r.ans.choice === r.item.gold) ok++;
            else if (kind === "score" && Math.round(r.ans.score) === r.item.gold) ok++;
            else if (kind === "noul" && (r.ans.noul >= 0.5) === r.item.gold) ok++;
        }
        console.log(`  [${task.padEnd(8)}] (${kind.padEnd(6)}) -> Aciertos: ${ok}/${records.length} (${((ok / records.length) * 100).toFixed(1)}%)`);
    }
}

main().catch(err => {
    console.error(err);
    process.exit(1);
});
