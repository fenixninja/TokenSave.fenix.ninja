/**
 * bench-web.js - Ejecutor de Benchmarks v2 en Navegador (WebGPU/WASM) para TokenSafe Remaster
 * Usa el motor unificado js/decision-engine.js (50 mejoras) + js/model.js (Qwen3-Embedding-0.6B-ONNX)
 */

import { getEmbeddingModel, computeRawEmbeddings } from "./model.js";
import { evaluateSystemOne } from "./decision-engine.js";
import { formatCleanState } from "./text-preprocessor.js";

const TASKS = ["ag_news", "amazon", "boolq", "emotion", "irony", "sms_spam", "sst2", "sst5"];

const btnStart = document.getElementById("btn-start-bench");
const inpLimit = document.getElementById("inp-limit");
const selTask = document.getElementById("sel-task");
const elStatus = document.getElementById("bench-status");
const elLog = document.getElementById("bench-log");
const elSummary = document.getElementById("summary-container");

export async function evaluateRemasterItem(state, questionSpec) {
    const { answers } = await evaluateSystemOne(state, { q: questionSpec }, computeRawEmbeddings);
    return answers.q;
}

async function loadDatasetItems(selectedTask, limit) {
    const targetTasks = selectedTask === "all" ? TASKS : [selectedTask];
    const items = [];
    for (const task of targetTasks) {
        const resp = await fetch(`bench/data/${task}.jsonl`);
        const text = await resp.text();
        const lines = text.split("\n").filter(l => l.trim().length > 0).slice(0, limit);
        for (const line of lines) {
            const parsed = JSON.parse(line);
            parsed.task = task;
            items.push(parsed);
        }
    }
    return items;
}

btnStart.addEventListener("click", async () => {
    btnStart.disabled = true;
    elLog.innerHTML = "";
    elSummary.innerHTML = "";
    const limit = Math.max(1, parseInt(inpLimit.value || "5", 10));

    try {
        elStatus.textContent = "Cargando modelo Qwen3-Embedding-0.6B-ONNX...";
        await getEmbeddingModel((info) => {
            if (info.status === "progress") {
                elStatus.textContent = `Descargando/cargando modelo... ${Math.round(info.progress || 0)}%`;
            }
        });

        const items = await loadDatasetItems(selTask.value, limit);
        const byTask = {};
        let totalOk = 0;

        for (let i = 0; i < items.length; i++) {
            const it = items[i];
            elStatus.textContent = `Evaluando [${i + 1}/${items.length}] ${it.id}...`;
            const t0 = performance.now();
            const ans = await evaluateRemasterItem(it.state, it.questions.q);
            const dt = performance.now() - t0;

            let ok = false;
            let predStr = "";
            if (it.kind === "choice") {
                ok = ans.choice === it.gold;
                predStr = ans.choice;
            } else if (it.kind === "score") {
                ok = Math.round(ans.score) === it.gold;
                predStr = `${ans.score.toFixed(2)} (nivel=${Math.round(ans.score)})`;
            } else {
                ok = (ans.noul >= 0.5) === it.gold;
                predStr = `${ans.noul >= 0.5} (p=${ans.noul.toFixed(2)})`;
            }

            if (ok) totalOk++;
            if (!byTask[it.task]) byTask[it.task] = { kind: it.kind, ok: 0, n: 0, latencies: [] };
            byTask[it.task].n++;
            if (ok) byTask[it.task].ok++;
            byTask[it.task].latencies.push(dt);

            const stTxt = formatCleanState(it.state).replace(/\n/g, " ").slice(0, 90);
            const lineHtml = `[${i + 1}/${items.length}] [${it.task} | ${it.id}] <span class="${ok ? "ok" : "fail"}">${ok ? "✔ ACIERTO" : "✘ FALLO"}</span> | pred=${predStr} vs gold=${it.gold} | ${dt.toFixed(0)} ms\n    State: ${stTxt}...\n`;
            elLog.insertAdjacentHTML("beforeend", lineHtml);
            elLog.scrollTop = elLog.scrollHeight;
        }

        const pctTotal = items.length > 0 ? ((totalOk / items.length) * 100).toFixed(1) : "0.0";
        elStatus.textContent = `Completado: ${totalOk}/${items.length} (${pctTotal}%)`;

        let rowsHtml = "";
        for (const [task, stats] of Object.entries(byTask)) {
            const sortedLat = [...stats.latencies].sort((a, b) => a - b);
            const p50 = sortedLat[Math.floor(sortedLat.length / 2)] || 0;
            const pct = ((stats.ok / stats.n) * 100).toFixed(1);
            rowsHtml += `<tr><td><code>${task}</code></td><td><code>${stats.kind}</code></td><td>${stats.ok} / ${stats.n}</td><td><strong>${pct} %</strong></td><td>${p50.toFixed(0)} ms</td></tr>`;
        }
        rowsHtml += `<tr style="background:#1e293b;"><td><strong>TOTAL GLOBAL</strong></td><td><strong>Todas</strong></td><td><strong>${totalOk} / ${items.length}</strong></td><td><strong>${pctTotal} %</strong></td><td>—</td></tr>`;

        elSummary.innerHTML = `
            <table class="bench-summary-table">
                <thead><tr><th>Tarea</th><th>Tipo</th><th>Aciertos</th><th>Porcentaje</th><th>Latencia p50 (ms)</th></tr></thead>
                <tbody>${rowsHtml}</tbody>
            </table>
        `;
    } catch (err) {
        console.error(err);
        elStatus.textContent = `Error: ${err.message}`;
    } finally {
        btnStart.disabled = false;
    }
});
