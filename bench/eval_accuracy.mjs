#!/usr/bin/env node
/**
 * eval_accuracy.mjs - Suite de Evaluación y Benchmarks v2 para TokenSafe Remaster (50 Mejoras)
 * Usa js/decision-engine.js + js/primitives.js + js/text-preprocessor.js + Qwen3-Embedding-0.6B-ONNX (q4f16)
 * Incluye subcomandos 'summarize', 'compare' (#49) y flag '--explain-failures' (#50).
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pipeline, env } from "@huggingface/transformers";
import { cosineSimilarityRaw } from "../js/primitives.js";
import { formatCleanState } from "../js/text-preprocessor.js";
import { evaluateSystemOne } from "../js/decision-engine.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, "..");
const DATA_DIR = path.join(__dirname, "data");
const RESULTS_FILE = path.join(__dirname, "results", "accuracy.jsonl");
const CACHE_DIR = path.join(ROOT_DIR, ".cache", "huggingface");

env.cacheDir = CACHE_DIR;
if (env.backends && env.backends.onnx) {
    env.backends.onnx.logLevel = "error";
}

const MODEL_ID = "onnx-community/Qwen3-Embedding-0.6B-ONNX";
const MAX_CACHE_ENTRIES = 5000;
const embeddingCache = new Map();
let embedPipeline = null;

function leerMemDisponibleMb() {
    try {
        const meminfo = fs.readFileSync("/proc/meminfo", "utf8");
        const match = meminfo.match(/^MemAvailable:\s+(\d+)\s+kB/m);
        if (match) {
            return `${Math.round(Number(match[1]) / 1024)} MB`;
        }
    } catch (_) {}
    return "N/D";
}

function setLruCache(key, vec) {
    if (embeddingCache.has(key)) embeddingCache.delete(key);
    embeddingCache.set(key, vec);
    if (embeddingCache.size > MAX_CACHE_ENTRIES) {
        const oldest = embeddingCache.keys().next().value;
        embeddingCache.delete(oldest);
    }
}

export async function getRemasterModel() {
    if (embedPipeline) return embedPipeline;
    // #45: Configure 4 intra-op threads matching Intel Core i5-3470S physical cores
    embedPipeline = await pipeline("feature-extraction", MODEL_ID, {
        dtype: "q4f16",
        session_options: {
            logSeverityLevel: 3,
            intraOpNumThreads: 4,
            interOpNumThreads: 1
        }
    });
    return embedPipeline;
}

/**
 * #39 & #41: Single-batch raw embedding with direct Float32Array tensor extraction.
 */
export async function computeRawEmbeddingsNode(texts) {
    const model = await getRemasterModel();
    const missingIdx = [];
    const missingTexts = [];

    texts.forEach((t, idx) => {
        if (!embeddingCache.has(t)) {
            missingIdx.push(idx);
            missingTexts.push(t);
        }
    });

    if (missingTexts.length > 0) {
        const out = await model(missingTexts, { pooling: "mean", normalize: true });
        if (out && out.data && out.dims && out.dims.length >= 2) {
            const dim = out.dims[out.dims.length - 1];
            missingIdx.forEach((origIdx, i) => {
                const start = i * dim;
                setLruCache(texts[origIdx], new Float32Array(out.data.slice(start, start + dim)));
            });
        } else {
            const list = out.tolist();
            missingIdx.forEach((origIdx, i) => {
                setLruCache(texts[origIdx], new Float32Array(list[i]));
            });
        }
    }

    return texts.map(t => embeddingCache.get(t));
}

export async function computeEmbeddingsNode(texts, prefix = "Query:") {
    const formatted = texts.map(t => (prefix ? `${prefix} ${String(t).trim()}` : String(t).trim()));
    return computeRawEmbeddingsNode(formatted);
}

export function cosineSimilarity(vecA, vecB) {
    return cosineSimilarityRaw(vecA, vecB);
}

export function formatState(state) {
    return formatCleanState(state);
}

export async function systemOneRemaster(state, questions, returnDiagnostics = false) {
    const { answers, diagnostics } = await evaluateSystemOne(state, questions, computeRawEmbeddingsNode);
    return returnDiagnostics ? { answers, diagnostics } : answers;
}

// --- Métricas de evaluación ---
function mean(xs) {
    return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}

function pct(xs, p) {
    if (!xs.length) return 0;
    const s = [...xs].sort((a, b) => a - b);
    const idx = Math.min(s.length - 1, Math.round((p / 100) * (s.length - 1)));
    return s[idx];
}

function auroc(scores, labels) {
    const pos = scores.filter((_, i) => labels[i]);
    const neg = scores.filter((_, i) => !labels[i]);
    if (!pos.length || !neg.length) return NaN;
    let wins = 0;
    for (const p of pos) {
        for (const n of neg) {
            wins += p > n ? 1.0 : p === n ? 0.5 : 0.0;
        }
    }
    return wins / (pos.length * neg.length);
}

function spearman(a, b) {
    const ranks = (x) => {
        const order = x.map((_, i) => i).sort((i, j) => x[i] - x[j]);
        const r = new Array(x.length).fill(0);
        let i = 0;
        while (i < order.length) {
            let j = i;
            while (j + 1 < order.length && x[order[j + 1]] === x[order[i]]) j++;
            for (let k = i; k <= j; k++) r[order[k]] = (i + j) / 2 + 1;
            i = j + 1;
        }
        return r;
    };
    const ra = ranks(a);
    const rb = ranks(b);
    const ma = mean(ra);
    const mb = mean(rb);
    let cov = 0, va = 0, vb = 0;
    for (let i = 0; i < ra.length; i++) {
        cov += (ra[i] - ma) * (rb[i] - mb);
        va += (ra[i] - ma) ** 2;
        vb += (rb[i] - mb) ** 2;
    }
    return va && vb ? cov / (Math.sqrt(va) * Math.sqrt(vb)) : NaN;
}

function ece(conf, correct, bins = 10) {
    let tot = 0;
    for (let b = 0; b < bins; b++) {
        const lo = b / bins, hi = (b + 1) / bins;
        const idx = [];
        for (let i = 0; i < conf.length; i++) {
            if ((conf[i] >= lo && conf[i] < hi) || (b === bins - 1 && conf[i] === 1.0)) idx.push(i);
        }
        if (idx.length) {
            const acc = idx.reduce((s, i) => s + (correct[i] ? 1 : 0), 0) / idx.length;
            const avg = idx.reduce((s, i) => s + conf[i], 0) / idx.length;
            tot += (idx.length / conf.length) * Math.abs(acc - avg);
        }
    }
    return tot;
}

function macroF1(pred, gold) {
    const labels = Array.from(new Set([...gold, ...pred])).sort();
    const f1s = labels.map(label => {
        let tp = 0, fp = 0, fn = 0;
        for (let i = 0; i < pred.length; i++) {
            if (pred[i] === label && gold[i] === label) tp++;
            else if (pred[i] === label && gold[i] !== label) fp++;
            else if (pred[i] !== label && gold[i] === label) fn++;
        }
        const pr = tp + fp ? tp / (tp + fp) : 0;
        const rc = tp + fn ? tp / (tp + fn) : 0;
        return pr + rc ? (2 * pr * rc) / (pr + rc) : 0;
    });
    return mean(f1s);
}

export function taskMetrics(rows) {
    const kind = rows[0].kind;
    const m = {
        n: rows.length,
        p50_ms: Number(pct(rows.map(r => r.latency_ms), 50).toFixed(1)),
        tokens: Math.round(mean(rows.map(r => r.input_tokens || 0)))
    };
    if (kind === "choice") {
        const pred = rows.map(r => r.answer.choice);
        const gold = rows.map(r => r.gold);
        const correct = pred.map((p, i) => p === gold[i]);
        const pmax = rows.map(r => Math.max(...Object.values(r.answer.probabilities)));
        m.acc = mean(correct.map(Number));
        m.f1 = macroF1(pred, gold);
        m.ece = ece(pmax, correct);
        m.pmax = mean(pmax);
        m.aciertos = correct.filter(Boolean).length;
    } else if (kind === "score") {
        const s = rows.map(r => Number(r.answer.score));
        const g = rows.map(r => Number(r.gold));
        const argmax = rows.map(r => {
            const probs = r.answer.probabilities;
            return Number(Object.keys(probs).reduce((a, b) => (probs[a] >= probs[b] ? a : b)));
        });
        const mid = (Object.keys(rows[0].answer.legend).length - 1) / 2;
        const correct = s.map((a, i) => Math.round(a) === g[i]);
        m.mae = mean(s.map((a, i) => Math.abs(a - g[i])));
        m.acc_round = mean(correct.map(Number));
        m.acc_argmax = mean(argmax.map((a, i) => (a === g[i] ? 1 : 0)));
        m.spearman = spearman(s, g);
        m.mae_mid = mean(g.map(b => Math.abs(mid - b)));
        m.aciertos = correct.filter(Boolean).length;
    } else {
        const p = rows.map(r => Number(r.answer.noul));
        const y = rows.map(r => Boolean(r.gold));
        const pred = rows.map(r => r.answer.isTrue !== undefined ? Boolean(r.answer.isTrue) : (Number(r.answer.noul) >= (Number(r.answer.threshold) || 0.5)));
        const correct = pred.map((a, i) => a === y[i]);
        const conf = p.map((x, i) => (pred[i] ? x : 1 - x));
        m.auroc = auroc(p, y);
        m.acc = mean(correct.map(Number));
        m.brier = mean(p.map((x, i) => (x - (y[i] ? 1 : 0)) ** 2));
        m.ece = ece(conf, correct);
        m.mean_p = mean(p);
        m.aciertos = correct.filter(Boolean).length;
    }
    return m;
}

function quickSummary(rows, title = "TokenSafe Remaster v2 (50 Mejoras)") {
    const byTask = new Map();
    for (const r of rows) {
        if (!byTask.has(r.task)) byTask.set(r.task, []);
        byTask.get(r.task).push(r);
    }

    const sortedTasks = Array.from(byTask.keys()).sort();
    for (const t of sortedTasks) {
        const m = taskMetrics(byTask.get(t));
        const kv = Object.entries(m)
            .filter(([k]) => k !== "aciertos")
            .map(([k, v]) => (typeof v === "number" && !Number.isInteger(v) ? `${k}=${v.toFixed(3)}` : `${k}=${v}`))
            .join("  ");
        console.log(`  ${t.padEnd(10)} ${kv}`);
    }

    console.log(`\n### Tabla de Porcentajes — ${title}`);
    console.log("| Tarea | Tipo | Aciertos | Porcentaje | Latencia p50 (ms) |");
    console.log("|---|---|---:|---:|---:|");
    let totOk = 0, totN = 0;
    for (const t of sortedTasks) {
        const rs = byTask.get(t);
        const m = taskMetrics(rs);
        const kind = rs[0].kind;
        const accVal = kind === "score" ? m.acc_round : m.acc;
        totOk += m.aciertos;
        totN += m.n;
        console.log(`| \`${t}\` | \`${kind}\` | ${m.aciertos} / ${m.n} | **${(accVal * 100).toFixed(1)} %** | ${m.p50_ms.toFixed(0)} ms |`);
    }
    if (totN > 0) {
        console.log(`| **TOTAL GLOBAL** | **Todas** | **${totOk} / ${totN}** | **${((totOk / totN) * 100).toFixed(1)} %** | — |\n`);
    }
}

/**
 * #49: Compares two benchmark runs stored in accuracy.jsonl side-by-side.
 */
function compareRuns(allRows, runA = "tokensafe-remaster/qwen3-embedding-0.6b", runB = "tokensafe-remaster-v2/qwen3-embedding-0.6b") {
    const rowsA = allRows.filter(r => r.run === runA);
    const rowsB = allRows.filter(r => r.run === runB);
    if (!rowsA.length || !rowsB.length) {
        const available = Array.from(new Set(allRows.map(r => r.run)));
        console.log(`⚠ Runs disponibles en ${RESULTS_FILE}:`, available);
        return;
    }

    const byTaskA = new Map();
    const byTaskB = new Map();
    for (const r of rowsA) {
        if (!byTaskA.has(r.task)) byTaskA.set(r.task, []);
        byTaskA.get(r.task).push(r);
    }
    for (const r of rowsB) {
        if (!byTaskB.has(r.task)) byTaskB.set(r.task, []);
        byTaskB.get(r.task).push(r);
    }

    const tasks = Array.from(new Set([...byTaskA.keys(), ...byTaskB.keys()])).sort();
    console.log(`\n### Comparativa: Base (${runA}) vs Mejorado (${runB})`);
    console.log("| Tarea | Tipo | Base (%) | Mejorado v2 (%) | Delta (Δ) |");
    console.log("|---|---|---:|---:|---:|");

    let okA = 0, nA = 0, okB = 0, nB = 0;
    for (const t of tasks) {
        const rsA = byTaskA.get(t) || [];
        const rsB = byTaskB.get(t) || [];
        if (!rsA.length || !rsB.length) continue;
        const mA = taskMetrics(rsA);
        const mB = taskMetrics(rsB);
        const kind = rsB[0].kind;
        const pA = (kind === "score" ? mA.acc_round : mA.acc) * 100;
        const pB = (kind === "score" ? mB.acc_round : mB.acc) * 100;
        const delta = pB - pA;
        const sign = delta >= 0 ? "+" : "";
        okA += mA.aciertos; nA += mA.n;
        okB += mB.aciertos; nB += mB.n;
        console.log(`| \`${t}\` | \`${kind}\` | ${pA.toFixed(1)} % (${mA.aciertos}/${mA.n}) | **${pB.toFixed(1)} %** (${mB.aciertos}/${mB.n}) | **${sign}${delta.toFixed(1)} pp** |`);
    }
    if (nA > 0 && nB > 0) {
        const totPA = (okA / nA) * 100;
        const totPB = (okB / nB) * 100;
        const totDelta = totPB - totPA;
        const sign = totDelta >= 0 ? "+" : "";
        console.log(`| **TOTAL GLOBAL** | **Todas** | **${totPA.toFixed(1)} % (${okA}/${nA})** | **${totPB.toFixed(1)} % (${okB}/${nB})** | **${sign}${totDelta.toFixed(1)} pp** |\n`);
    }
}

function writeRows(rows, outPath = RESULTS_FILE) {
    if (!rows.length) return;
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    const run = rows[0].run;
    const tasks = new Set(rows.map(r => r.task));
    const kept = [];
    if (fs.existsSync(outPath)) {
        for (const line of fs.readFileSync(outPath, "utf8").split("\n")) {
            if (!line.trim()) continue;
            const r = JSON.parse(line);
            if (!(r.run === run && tasks.has(r.task))) kept.push(line);
        }
    }
    const allLines = [...kept, ...rows.map(r => JSON.stringify(r))];
    fs.writeFileSync(outPath, allLines.join("\n") + "\n", "utf8");
}

export function loadItems(tasks = null, limit = 5) {
    if (!fs.existsSync(DATA_DIR)) return [];
    const files = fs.readdirSync(DATA_DIR).filter(f => f.endsWith(".jsonl")).sort();
    const items = [];
    for (const f of files) {
        const task = path.basename(f, ".jsonl");
        if (tasks && tasks.length && !tasks.includes(task)) continue;
        const lines = fs.readFileSync(path.join(DATA_DIR, f), "utf8").split("\n").filter(l => l.trim());
        for (const line of lines.slice(0, limit)) {
            const parsed = JSON.parse(line);
            parsed.task = task;
            items.push(parsed);
        }
    }
    return items;
}

async function main() {
    const args = process.argv.slice(2);
    if (args[0] === "summarize") {
        if (!fs.existsSync(RESULTS_FILE)) {
            console.log(`⚠ No existe ${RESULTS_FILE}. Ejecuta primero el benchmark.`);
            return;
        }
        const allRows = fs.readFileSync(RESULTS_FILE, "utf8").split("\n").filter(l => l.trim()).map(l => JSON.parse(l));
        const byRun = new Map();
        for (const r of allRows) {
            if (!byRun.has(r.run)) byRun.set(r.run, []);
            byRun.get(r.run).push(r);
        }
        for (const [rName, rs] of byRun.entries()) {
            console.log(`=== Run: ${rName} (${rs.length} items) ===`);
            quickSummary(rs, rName);
        }
        return;
    }

    if (args[0] === "compare") {
        if (!fs.existsSync(RESULTS_FILE)) {
            console.log(`⚠ No existe ${RESULTS_FILE}.`);
            return;
        }
        const allRows = fs.readFileSync(RESULTS_FILE, "utf8").split("\n").filter(l => l.trim()).map(l => JSON.parse(l));
        compareRuns(allRows, args[1], args[2]);
        return;
    }

    let limit = 5;
    let tasks = null;
    let runName = "tokensafe-remaster-v3/qwen3-embedding-0.6b";
    let explainFailures = false;

    for (let i = 0; i < args.length; i++) {
        if (args[i] === "--limit" && args[i + 1]) {
            limit = parseInt(args[++i], 10);
        } else if (args[i] === "--name" && args[i + 1]) {
            runName = args[++i];
        } else if (args[i] === "--explain-failures") {
            explainFailures = true;
        } else if (args[i] === "--tasks") {
            tasks = [];
            while (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                tasks.push(args[++i]);
            }
        }
    }

    const items = loadItems(tasks, limit);
    if (!items.length) {
        console.error(`⚠ No se encontraron items en ${DATA_DIR}`);
        process.exit(1);
    }

    console.log(`Cargando modelo ${MODEL_ID} (q4f16, 4 hilos CPU)...`);
    await getRemasterModel();
    console.log(`✔ Modelo listo. Evaluando ${items.length} pruebas con motor v3 (calibración de umbrales + mode blending + BoolQ acelerado)...\n`);

    const rows = [];
    for (let i = 0; i < items.length; i++) {
        const it = items[i];
        const t0 = performance.now();
        const { answers, diagnostics } = await systemOneRemaster(it.state, it.questions, true);
        const dt = performance.now() - t0;
        const ans = answers.q;

        rows.push({
            run: runName,
            system: "tokensafe_remaster_v2",
            variant: "qwen3-embedding-0.6b-v2",
            task: it.task,
            kind: it.kind,
            id: it.id,
            gold: it.gold,
            answer: ans,
            latency_ms: Number(dt.toFixed(2)),
            input_tokens: Math.ceil(formatState(it.state).length / 4),
            response_model: MODEL_ID
        });

        let ok = false;
        let predStr = "";
        if (it.kind === "choice") {
            ok = ans.choice === it.gold;
            predStr = String(ans.choice);
        } else if (it.kind === "score") {
            ok = Math.round(ans.score) === it.gold;
            predStr = `${ans.score.toFixed(2)} (nivel=${Math.round(ans.score)}, am=${ans.argmaxLevel ?? 'N/A'})`;
        } else {
            const isDecisionTrue = ans.isTrue !== undefined ? Boolean(ans.isTrue) : (ans.noul >= (ans.threshold || 0.5));
            ok = isDecisionTrue === it.gold;
            predStr = `${isDecisionTrue} (p=${ans.noul.toFixed(2)}, th=${(ans.threshold ?? 0.5).toFixed(2)})`;
        }

        const icono = ok ? "✔ ACIERTO" : "✘ FALLO";
        let stTxt = formatState(it.state).replace(/\n/g, " ");
        if (stTxt.length > 90) stTxt = stTxt.slice(0, 87) + "...";

        console.log(
            `[${rows.length}/${items.length}] [${it.task} | ${it.id}] ${icono} | ` +
            `pred=${predStr} vs gold=${it.gold} | ${dt.toFixed(0)} ms | RAM libre: ${leerMemDisponibleMb()}\n` +
            `    State: ${stTxt}`
        );

        // #50: Explain failures diagnostic mode
        if (!ok && explainFailures && diagnostics && diagnostics.q) {
            console.log(`    [Diagnóstico #50]: ${JSON.stringify(diagnostics.q)}`);
        }
    }

    const aciertosTot = rows.filter(r =>
        (r.kind === "choice" && r.answer.choice === r.gold) ||
        (r.kind === "score" && Math.round(r.answer.score) === r.gold) ||
        (r.kind === "noul" && (r.answer.isTrue !== undefined ? Boolean(r.answer.isTrue) : (r.answer.noul >= (r.answer.threshold || 0.5))) === r.gold)
    ).length;
    const pctTot = ((aciertosTot / rows.length) * 100).toFixed(1);

    console.log(`\n${runName}: ${rows.length}/${items.length} done, 0 errors`);
    console.log(`--> Precisión global exacta: ${aciertosTot}/${rows.length} (${pctTot}%)\n`);

    writeRows(rows);
    quickSummary(rows);

    // Si existen corridas anteriores en accuracy.jsonl, mostrar la comparativa automática (v2 vs v3 o base vs v3)
    if (fs.existsSync(RESULTS_FILE)) {
        const allSaved = fs.readFileSync(RESULTS_FILE, "utf8").split("\n").filter(l => l.trim()).map(l => JSON.parse(l));
        const hasV2 = allSaved.some(r => r.run === "tokensafe-remaster-v2/qwen3-embedding-0.6b");
        const baselineToCompare = hasV2 ? "tokensafe-remaster-v2/qwen3-embedding-0.6b" : "tokensafe-remaster/qwen3-embedding-0.6b";
        if (runName !== baselineToCompare) {
            compareRuns(allSaved, baselineToCompare, runName);
        }
    }
}

if (process.argv[1] && path.resolve(process.argv[1]) === __filename) {
    main().catch(err => {
        console.error(err);
        process.exit(1);
    });
}
