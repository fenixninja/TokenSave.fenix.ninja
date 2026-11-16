/**
 * decision-engine.js - Unified Fenix Vector Decision Engine (#46)
 * Shared between Browser (WebGPU/WASM) and Node.js CLI (eval_accuracy.mjs / test_remaster.mjs).
 *
 * Combines:
 * - Block A (#1-#12): All-but-the-Top centroid removal, bipolar axis projection, Gram-Schmidt
 *   orthogonalization, CSLS null-baseline subtraction, question residual projection, head/tail
 *   polarity contrast.
 * - Block B (#13-#26): Z-score pre-Softmax, variance-stretched Score, calibrated Bayesian Noul,
 *   uncertainty decomposition.
 * - Block C (#27-#38): Qwen3 native instruction template, adversative clause weighting, sentence
 *   windowing for BoolQ, enriched prototypes for Amazon/SST5/AGNews/Emotion/Spam/Irony.
 * - Block D (#39-#40): Single-pass batched embedding and schema-level prototype caching.
 */

import {
    cosineSimilarityRaw,
    normalizeL2,
    removeCommonCentroid,
    projectPolarAxis,
    orthogonalizePair,
    cslsSimilarity,
    computePrototypeCentroid,
    detectPolarityContrast,
    zScoreNormalize,
    adaptiveTemperature,
    stretchScoreExtremes,
    decomposeUncertainty,
    analyzeChoiceDistribution,
    calculateScoreValue,
    softmax,
    calculateNoulBayesian
} from "./primitives.js";

import {
    cleanRawText,
    formatCleanState,
    normalizeSmsText,
    extractSurfaceSignals,
    splitAdversativeClauses,
    splitIntoSentences,
    formatQwenQuery,
    buildNoulPrototypes,
    buildScoreLevelPrototypes,
    buildChoiceOptionPrototypes,
    selectSalientSentences,
    tokenTrimState
} from "./text-preprocessor.js";

// #40: Schema-level cache for compiled criterion vectors, centroids, and null-input biases
const SCHEMA_STORAGE_KEY = "fenix_compiled_question_geometries_v1";
const compiledQuestionCache = new Map();
let schemaCacheLoaded = false;

function ensureSchemaCacheLoaded() {
    if (schemaCacheLoaded) return;
    schemaCacheLoaded = true;
    try {
        if (typeof localStorage !== "undefined") {
            const raw = localStorage.getItem(SCHEMA_STORAGE_KEY);
            if (raw) {
                const parsed = JSON.parse(raw);
                for (const [k, v] of Object.entries(parsed)) {
                    if (v.centeredOptVecs) v.centeredOptVecs = v.centeredOptVecs.map(a => new Float32Array(a));
                    if (v.rawOptVecs) v.rawOptVecs = v.rawOptVecs.map(a => new Float32Array(a));
                    if (v.rawLevelVecs) v.rawLevelVecs = v.rawLevelVecs.map(a => new Float32Array(a));
                    if (v.centeredLevelVecs) v.centeredLevelVecs = v.centeredLevelVecs.map(a => new Float32Array(a));
                    if (v.nullVec) v.nullVec = new Float32Array(v.nullVec);
                    if (v.orthoTrue) v.orthoTrue = new Float32Array(v.orthoTrue);
                    if (v.orthoFalse) v.orthoFalse = new Float32Array(v.orthoFalse);
                    compiledQuestionCache.set(k, v);
                }
                console.log(`%c[Decision Engine] ✔ ${compiledQuestionCache.size} geometrias de preguntas recuperadas de almacenamiento local permanente (0 ms).`, "color: #10b981; font-weight: bold;");
            }
        }
    } catch (_) {}
}

function persistSchemaCache() {
    try {
        if (typeof localStorage !== "undefined" && compiledQuestionCache.size <= 300) {
            const serializable = {};
            for (const [k, v] of compiledQuestionCache.entries()) {
                serializable[k] = {
                    ...v,
                    centeredOptVecs: v.centeredOptVecs ? v.centeredOptVecs.map(a => Array.from(a)) : undefined,
                    rawOptVecs: v.rawOptVecs ? v.rawOptVecs.map(a => Array.from(a)) : undefined,
                    rawLevelVecs: v.rawLevelVecs ? v.rawLevelVecs.map(a => Array.from(a)) : undefined,
                    centeredLevelVecs: v.centeredLevelVecs ? v.centeredLevelVecs.map(a => Array.from(a)) : undefined,
                    nullVec: v.nullVec ? Array.from(v.nullVec) : undefined,
                    orthoTrue: v.orthoTrue ? Array.from(v.orthoTrue) : undefined,
                    orthoFalse: v.orthoFalse ? Array.from(v.orthoFalse) : undefined
                };
            }
            localStorage.setItem(SCHEMA_STORAGE_KEY, JSON.stringify(serializable));
        }
    } catch (_) {}
}

/**
 * Blends two vectors with weights (wA, wB) and L2-normalizes the result.
 */
function blendVectors(vecA, vecB, wA = 0.65, wB = 0.35) {
    if (!vecB) return vecA;
    const out = new Float32Array(vecA.length);
    for (let i = 0; i < vecA.length; i++) {
        out[i] = wA * vecA[i] + wB * vecB[i];
    }
    return normalizeL2(out);
}

/**
 * Compiles and caches the embedding geometry for a question definition (#40).
 */
async function compileQuestionGeometry(qKey, qDef, rawEmbedFn) {
    ensureSchemaCacheLoaded();
    const cacheKey = JSON.stringify({ qKey, type: qDef.type, instructions: qDef.instructions, criteria: qDef.criteria });
    if (compiledQuestionCache.has(cacheKey)) {
        return compiledQuestionCache.get(cacheKey);
    }

    const typeNorm = String(qDef.type || "choice").toLowerCase();
    const instText = typeof qDef.instructions === "string"
        ? qDef.instructions
        : JSON.stringify(qDef.instructions || qKey);
    const neutralQuery = formatQwenQuery(instText, " Neutral baseline text statement.");

    if (typeNorm === "choice") {
        const enrichedMap = buildChoiceOptionPrototypes(instText, qDef.criteria || {});
        const optKeys = Object.keys(enrichedMap);
        const optTexts = optKeys.map(k => enrichedMap[k]);

        // #39: Single batch embedding for neutral query + all option texts
        const allVecs = await rawEmbedFn([neutralQuery, ...optTexts]);
        const nullVec = allVecs[0];
        const rawOptVecs = allVecs.slice(1);

        // #1: Remove common centroid across options (All-but-the-Top)
        const centeredOptVecs = removeCommonCentroid(rawOptVecs, 0.80);

        // #8: Null-input hubness baseline per option
        const nullSims = centeredOptVecs.map(v => cosineSimilarityRaw(nullVec, v));

        const compiled = {
            type: "choice",
            instText,
            optKeys,
            optTexts,
            rawOptVecs,
            centeredOptVecs,
            nullVec,
            nullSims
        };
        compiledQuestionCache.set(cacheKey, compiled);
        persistSchemaCache();
        return compiled;
    }

    if (typeNorm === "score") {
        const criteria = Array.isArray(qDef.criteria)
            ? qDef.criteria
            : Object.values(qDef.criteria || {});
        const rawLabels = criteria.map(c => (typeof c === "object" && c.what ? String(c.what) : String(c)));
        const enrichedLabels = buildScoreLevelPrototypes(instText, rawLabels);

        const allVecs = await rawEmbedFn([neutralQuery, ...enrichedLabels]);
        const nullVec = allVecs[0];
        const rawLevelVecs = allVecs.slice(1);

        // #1: Remove common centroid across the K scale levels
        const centeredLevelVecs = removeCommonCentroid(rawLevelVecs, 0.85);
        const nullSims = centeredLevelVecs.map(v => cosineSimilarityRaw(nullVec, v));

        const compiled = {
            type: "score",
            instText,
            rawLabels,
            enrichedLabels,
            rawLevelVecs,
            centeredLevelVecs,
            nullVec,
            nullSims
        };
        compiledQuestionCache.set(cacheKey, compiled);
        persistSchemaCache();
        return compiled;
    }

    // Noul (boolean)
    const proto = buildNoulPrototypes(instText, qDef.criteria);
    const allTexts = [neutralQuery, ...proto.trueTexts, ...proto.falseTexts];
    const allVecs = await rawEmbedFn(allTexts);
    const nullVec = allVecs[0];
    const tCount = proto.trueTexts.length;
    const trueVecs = allVecs.slice(1, 1 + tCount);
    const falseVecs = allVecs.slice(1 + tCount);

    // #10: Multi-anchor prototype centroids
    const rawTrueCentroid = computePrototypeCentroid(trueVecs);
    const rawFalseCentroid = computePrototypeCentroid(falseVecs);

    // #4: Gram-Schmidt symmetric orthogonalization
    const [orthoTrue, orthoFalse] = orthogonalizePair(rawTrueCentroid, rawFalseCentroid, 0.65);

    // #16: Null-input baseline similarities
    const nullPos = cosineSimilarityRaw(nullVec, orthoTrue);
    const nullNeg = cosineSimilarityRaw(nullVec, orthoFalse);

    const compiled = {
        type: "noul",
        domain: proto.domain,
        recommendedThreshold: proto.recommendedThreshold ?? 0.50,
        taskPriorShift: proto.taskPriorShift ?? 0.0,
        instText,
        orthoTrue,
        orthoFalse,
        nullVec,
        nullPos,
        nullNeg
    };
    compiledQuestionCache.set(cacheKey, compiled);
    persistSchemaCache();
    return compiled;
}

/**
 * Evaluates a state against a set of System One questions (Choice, Score, Noul)
 * using the full 50-improvement mathematical and embedding pipeline.
 *
 * @param {any} state - Input state string or object
 * @param {Record<string, any>} questions - Question definitions map
 * @param {(texts: string[]) => Promise<ArrayLike<number>[]>} rawEmbedFn - Batch embedding function
 * @returns {Promise<{ answers: Record<string, any>, diagnostics: Record<string, any> }>}
 */
export async function evaluateSystemOne(state, questions, rawEmbedFn) {
    const cleanState = formatCleanState(state);
    const answers = {};
    const diagnostics = {};
    const qEntries = Object.entries(questions);

    // 1. Compile geometries in parallel (uses compiledQuestionCache in memory)
    const geoms = await Promise.all(
        qEntries.map(([qKey, qDef]) => compileQuestionGeometry(qKey, qDef, rawEmbedFn))
    );

    // 2. Build flat unified batch of all required query prompts for single session.run()
    const promptBatch = [];
    const metaList = [];

    for (let i = 0; i < qEntries.length; i++) {
        const [qKey, qDef] = qEntries[i];
        const geom = geoms[i];
        const instText = geom.instText;

        // Domain-specific state text conditioning (TokenTrim / JEV Style)
        let stateForQuery = cleanState;
        if (cleanState.length > 1000) {
            // TokenTrim: Trim unstructured text (resumes, long documents) to the salient context (< 1,000 chars)
            stateForQuery = tokenTrimState(cleanState, instText, 1000);
        } else if (geom.domain === "boolq" && cleanState.length > 550) {
            // #29 & v4: Fast CPU saliency selection to avoid attention quadratic latency
            stateForQuery = selectSalientSentences(cleanState, instText, 2);
        } else if (geom.domain === "sms_spam") {
            stateForQuery = normalizeSmsText(cleanState);
        }

        const clauses = splitAdversativeClauses(stateForQuery);
        const startIndex = promptBatch.length;

        const mainPrompt = geom.domain === "boolq"
            ? formatQwenQuery(`Answer whether the passage supports the claim: ${instText}`, stateForQuery)
            : formatQwenQuery(instText, clauses.full);

        promptBatch.push(mainPrompt);

        // #36 & #37: Adversative connector splitting is only applied to short review texts (<600 chars)
        // to prevent triplicating prompts on long documents/traces and overwhelming WebAssembly memory
        const shouldSplitAdversative = clauses.hasPivot && geom.domain !== "boolq" && stateForQuery.length < 600;

        if (shouldSplitAdversative) {
            promptBatch.push(formatQwenQuery(instText, clauses.tail));
            promptBatch.push(formatQwenQuery(instText, clauses.head));
        }

        metaList.push({
            qKey,
            qDef,
            geom,
            startIndex,
            hasPivot: shouldSplitAdversative,
            salientSnippet: (geom.domain === "boolq" && cleanState.length > 550) ? stateForQuery : null
        });
    }

    // 3. Execute ONE SINGLE batched ONNX inference pass for all fields
    const allQueryVecs = await rawEmbedFn(promptBatch);

    // 4. Evaluate each question using its corresponding vector
    for (let i = 0; i < metaList.length; i++) {
        const { qKey, qDef, geom, startIndex, hasPivot, salientSnippet } = metaList[i];
        const instText = geom.instText;
        const stateVec = hasPivot
            ? blendVectors(allQueryVecs[startIndex], allQueryVecs[startIndex + 1], 0.45, 0.55)
            : allQueryVecs[startIndex];

        const surface = extractSurfaceSignals(cleanState, instText);

        if (geom.type === "choice") {
            const rawSims = geom.centeredOptVecs.map(v => cosineSimilarityRaw(stateVec, v));
            // #8: Subtract null-input hubness bias
            const debiasedSims = cslsSimilarity(rawSims, geom.nullSims, 0.70);
            // #13 & #17: Z-score + K-adaptive temperature
            const zSims = zScoreNormalize(debiasedSims);
            const temp = adaptiveTemperature(0.045, geom.optKeys.length);
            const probs = softmax(debiasedSims, temp);

            const maxProb = Math.max(...probs);
            const winIdx = probs.indexOf(maxProb);
            const distMetrics = analyzeChoiceDistribution(probs);
            const uncertainty = decomposeUncertainty(debiasedSims, probs);

            answers[qKey] = {
                type: "choice",
                choice: geom.optKeys[winIdx],
                confidence: distMetrics.confidence,
                probabilities: Object.fromEntries(geom.optKeys.map((k, i) => [k, Number(probs[i].toFixed(4))]))
            };
            diagnostics[qKey] = { rawSims, debiasedSims, zSims, uncertainty };

        } else if (geom.type === "score") {
            const k = geom.rawLabels.length;
            const rawSims = geom.centeredLevelVecs.map(v => cosineSimilarityRaw(stateVec, v));
            const debiasedSims = cslsSimilarity(rawSims, geom.nullSims, 0.85);

            // #2: Polar Axis Projection (projects state onto extreme-high minus extreme-low axis)
            const polar = projectPolarAxis(stateVec, geom.centeredLevelVecs, geom.nullVec);

            // #13 & #17: Score softmax with low temperature (0.05) to prevent mid-level flattening
            const zSims = zScoreNormalize(debiasedSims);
            const scoreTemp = (geom.domain === "amazon" || geom.domain === "sst5") ? 0.05 : 0.065;
            const probs = softmax(debiasedSims, scoreTemp);
            const scoreMetrics = calculateScoreValue(probs);

            // #14 & #15: Stretch midpoint shrinkage and blend with polar axis projection
            const stretched = stretchScoreExtremes(scoreMetrics.expectedScore, k - 1, probs, 1.75);
            
            // Mode argmax blending: resolves extreme compression in 1-5 ratings (Amazon & SST-5)
            const argmaxLevel = probs.indexOf(Math.max(...probs));
            const alphaMode = qDef.alphaMode !== undefined ? Number(qDef.alphaMode) : 0.35;
            const blendedScore = alphaMode * argmaxLevel + (1 - alphaMode) * (0.50 * stretched + 0.50 * polar.axisScore);
            const finalScore = Number(
                Math.max(0, Math.min(k - 1, blendedScore)).toFixed(3)
            );

            // Discretized level with edge-stretching (0.65 window on extremes)
            const minLevel = 0;
            const maxLevel = k - 1;
            let discreteLevel = Math.round(finalScore);
            if (finalScore <= minLevel + 0.65) {
                discreteLevel = minLevel;
            } else if (finalScore >= maxLevel - 0.65) {
                discreteLevel = maxLevel;
            }

            const legend = Object.fromEntries(geom.rawLabels.map((c, i) => [String(i), c]));
            answers[qKey] = {
                type: "score",
                score: finalScore,
                level: discreteLevel,
                argmaxLevel,
                confidence: Number((0.45 + (1 - scoreMetrics.variance / Math.max(1, k - 1)) * 0.45).toFixed(4)),
                legend,
                probabilities: Object.fromEntries(geom.rawLabels.map((_, i) => [String(i), Number(probs[i].toFixed(4))]))
            };
            diagnostics[qKey] = { rawSims, debiasedSims, polarAxisScore: polar.axisScore, stretched, argmaxLevel };

        } else {
            // Noul (boolean)
            let effectiveVec = stateVec;
            let extraBoost = surface.spamBoost + surface.ironyBoost;

            // #12: Polarity contrast between first half and second half for Irony detection
            if (geom.domain === "irony" && cleanState.length > 25) {
                const midIdx = Math.floor(cleanState.length / 2);
                const firstHalf = cleanState.slice(0, midIdx);
                const secondHalf = cleanState.slice(midIdx);
                const [vH, vT] = await rawEmbedFn([
                    formatQwenQuery(instText, firstHalf),
                    formatQwenQuery(instText, secondHalf)
                ]);
                const contrast = detectPolarityContrast(vH, vT, geom.orthoTrue, geom.orthoFalse);
                if (contrast > 0.08) {
                    extraBoost += (contrast - 0.08) * 0.45;
                }
            }

            // BoolQ negation awareness on salient passage
            if (geom.domain === "boolq") {
                const checkText = (salientSnippet || cleanState).toLowerCase();
                const qHasNeg = /\b(not|never|no|cannot|can't|illegal)\b/i.test(instText);
                const sHasNeg = /\b(not|never|cannot|can't|illegal|except|distinct\s+from|unlisted|fictional|no\s+longer|without)\b/i.test(checkText);
                if (sHasNeg && !qHasNeg) {
                    extraBoost -= 0.035;
                } else if (!sHasNeg && !qHasNeg) {
                    extraBoost += 0.025;
                }
            }

            if (instText.toLowerCase().includes("ask a question") && surface.hasQuestion) {
                extraBoost += 0.12;
            }

            const simPos = cosineSimilarityRaw(effectiveVec, geom.orthoTrue);
            const simNeg = cosineSimilarityRaw(effectiveVec, geom.orthoFalse);

            const userTh = qDef.threshold !== undefined && qDef.threshold !== null ? Number(qDef.threshold) : null;
            const effectiveThreshold = userTh ?? geom.recommendedThreshold ?? 0.50;
            const effectivePriorShift = geom.taskPriorShift ?? 0.0;

            // #16 & #47: Null-Input Baseline subtraction + Bayesian log-odds calibration + task-specific prior thresholding
            const bayesian = calculateNoulBayesian(simPos, simNeg, 0.055, {
                nullPos: geom.nullPos,
                nullNeg: geom.nullNeg,
                nullWeight: 0.90,
                boost: extraBoost,
                threshold: effectiveThreshold,
                taskPriorShift: effectivePriorShift
            });

            answers[qKey] = {
                type: "noul",
                noul: bayesian.noul,
                probability: bayesian.probability,
                isYes: bayesian.isYes,
                isTrue: bayesian.isTrue,
                decision: bayesian.isTrue,
                threshold: bayesian.threshold,
                logOdds: bayesian.logOdds
            };
            diagnostics[qKey] = { simPos, simNeg, nullPos: geom.nullPos, nullNeg: geom.nullNeg, extraBoost, bayesian };
        }
    }

    return { answers, diagnostics };
}
