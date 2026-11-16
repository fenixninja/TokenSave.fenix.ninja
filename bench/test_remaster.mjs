#!/usr/bin/env node
/**
 * test_remaster.mjs - Tests Unitarios (js/primitives.js) y de Integración System One (Qwen3-Embedding-0.6B-ONNX)
 */

import assert from "node:assert/strict";
import {
    softmax,
    calculateChoiceConfidence,
    analyzeChoiceDistribution,
    calculateScoreValue,
    calculateNoulBayesian,
    calculateCompositeScore
} from "../js/primitives.js";
import { systemOneRemaster } from "./eval_accuracy.mjs";

async function runUnitTests() {
    console.log("=== 1. Tests Unitarios de Primitivas Matemáticas (js/primitives.js) ===");

    const probs = softmax([0.82, 0.71, 0.65], 0.05);
    const sum = probs.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1.0) < 1e-6, "Softmax debe sumar 1.0");
    assert.ok(probs[0] > 0.80, "Softmax con T=0.05 debe amplificar claramente al ganador");
    console.log("  ✔ softmax(T=0.05) verificado:", probs.map(p => p.toFixed(3)));

    const dist = analyzeChoiceDistribution(probs);
    assert.ok(dist.confidence > 0.7 && dist.marginDelta > 0.5, "Confianza y margen deben ser altos");
    console.log("  ✔ analyzeChoiceDistribution verificado:", dist);

    const scoreMetrics = calculateScoreValue([0.05, 0.15, 0.80]);
    assert.ok(scoreMetrics.expectedScore > 1.6 && scoreMetrics.isUnimodal, "Expected score debe acercarse a 2");
    console.log("  ✔ calculateScoreValue verificado:", scoreMetrics);

    const noul = calculateNoulBayesian(0.84, 0.62, 0.08);
    assert.equal(noul.isYes, true);
    assert.ok(noul.probability > 0.90);
    console.log("  ✔ calculateNoulBayesian verificado:", noul);

    const mcda = calculateCompositeScore({ a: 0.9, b: 0.8 }, { a: 2, b: 1 });
    assert.ok(mcda.compositeScore > 0.85 && mcda.topsisScore > 0.80);
    console.log("  ✔ calculateCompositeScore (MCDA / TOPSIS) verificado:", mcda.percentage, "\n");
}

async function runSystemOneIntegrationTests() {
    console.log("=== 2. Tests de Integración System One (Qwen3-Embedding-0.6B-ONNX) ===\n");

    // Caso 1: Billing + Tone + Urgency
    const t0 = performance.now();
    const res1 = await systemOneRemaster(
        "I was charged twice. Please help ASAP.",
        {
            billing: {
                type: "noul",
                instructions: "Is this about billing?",
                criteria: {
                    true: "The message is about billing, duplicate charges, payments, or invoices.",
                    false: "The message is unrelated to billing or payments."
                }
            },
            tone: {
                type: "choice",
                instructions: "What is the tone?",
                criteria: {
                    calm: "Calm, relaxed, and unhurried communication.",
                    angry: "Hostile, urgent, upset, or demanding immediate resolution."
                }
            },
            urgency: {
                type: "score",
                instructions: "How urgent is this?",
                criteria: [
                    "Low urgency: routine inquiry with no time pressure.",
                    "Medium urgency: standard issue needing timely response.",
                    "High urgency: critical issue requiring immediate assistance ASAP."
                ]
            }
        }
    );
    console.log(`[Test 1] Billing + Tone + Urgency (${(performance.now() - t0).toFixed(0)} ms):`);
    console.log("  ->", JSON.stringify(res1));
    assert.ok(res1.billing.noul >= 0.5, "billing debe ser >= 0.5");
    assert.equal(res1.tone.choice, "angry", "tone debe ser angry");
    assert.ok(Math.round(res1.urgency.score) >= 1, "urgency redondeado debe ser >= 1 (medium/high)");
    console.log("  ✔ Test 1 PASADO\n");

    // Caso 2: Safari Bug Severity
    const t1 = performance.now();
    const res2 = await systemOneRemaster(
        "The export button crashes in Safari.",
        {
            severity: {
                type: "score",
                instructions: "How severe is this software bug?",
                criteria: [
                    "Cosmetic: minor visual or styling glitch",
                    "Degraded: feature works slowly",
                    "Blocking: complete crash of the feature"
                ]
            }
        }
    );
    console.log(`[Test 2] Safari Export Bug Severity (${(performance.now() - t1).toFixed(0)} ms):`);
    console.log("  ->", JSON.stringify(res2));
    assert.ok(Math.round(res2.severity.score) >= 1, "severity redondeado debe ser >= 1 (degraded/blocking)");
    console.log("  ✔ Test 2 PASADO\n");

    // Caso 3: Multi-Question Routing
    const t2 = performance.now();
    const res3 = await systemOneRemaster(
        "Hi, our engineering team of 40 needs enterprise pricing and SSO integration. Can you send a quote?",
        {
            team: {
                type: "choice",
                instructions: "Which team should handle this text?",
                criteria: {
                    billing: "Payments and invoices",
                    technical: "Software bugs and crashes",
                    sales: "Pricing, enterprise licenses, quotes, and commercial plans"
                }
            },
            asks: {
                type: "noul",
                instructions: "Does the text ask a question?",
                criteria: {
                    true: "The writer asks a question or requests a quote.",
                    false: "Statement with no question."
                }
            }
        }
    );
    console.log(`[Test 3] Multi-Question Routing (${(performance.now() - t2).toFixed(0)} ms):`);
    console.log("  ->", JSON.stringify(res3));
    assert.equal(res3.team.choice, "sales", "team debe ser sales");
    assert.ok(res3.asks.noul >= 0.5, "asks debe ser >= 0.5");
    console.log("  ✔ Test 3 PASADO\n");

    console.log("✅ TODOS LOS TESTS DE TOKENSAFE REMASTER PASARON CORRECTAMENTE.");
}

async function main() {
    await runUnitTests();
    await runSystemOneIntegrationTests();
}

main().catch(err => {
    console.error("❌ Fallo en tests:", err);
    process.exit(1);
});
