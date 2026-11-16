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
    buildChoiceOptionPrototypes
} from "./text-preprocessor.js";

// #40: Schema-level cache for compiled criterion vectors, centroids, and null-input biases
const compiledQuestionCache = new Map();

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
        instText,
        orthoTrue,
        orthoFalse,
        nullVec,
        nullPos,
        nullNeg
    };
    compiledQuestionCache.set(cacheKey, compiled);
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

    for (const [qKey, qDef] of Object.entries(questions)) {
        const geom = await compileQuestionGeometry(qKey, qDef, rawEmbedFn);
        const instText = geom.instText;
        const surface = extractSurfaceSignals(cleanState, instText);

        // #36 & #37: Adversative clause segmentation (gives higher weight to post-"but" clause)
        const textForQuery = geom.domain === "sms_spam" ? normalizeSmsText(cleanState) : cleanState;
        const clauses = splitAdversativeClauses(textForQuery);

        const queryPrompts = [formatQwenQuery(instText, clauses.full)];
        if (clauses.hasPivot) {
            queryPrompts.push(formatQwenQuery(instText, clauses.tail));
            queryPrompts.push(formatQwenQuery(instText, clauses.head));
        }

        const queryVecs = await rawEmbedFn(queryPrompts);
        const stateVec = clauses.hasPivot
            ? blendVectors(queryVecs[0], queryVecs[1], 0.45, 0.55)
            : queryVecs[0];

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

            // #13 & #17: Combine debiased similarity with Z-score scaled softmax
            const zSims = zScoreNormalize(debiasedSims);
            const probs = softmax(zSims, 0.65);
            const scoreMetrics = calculateScoreValue(probs);

            // #14 & #15: Stretch midpoint shrinkage and blend with polar axis projection
            const stretched = stretchScoreExtremes(scoreMetrics.expectedScore, k - 1, probs, 1.75);
            const finalScore = Number(
                Math.max(0, Math.min(k - 1, 0.55 * stretched + 0.45 * polar.axisScore)).toFixed(3)
            );

            const legend = Object.fromEntries(geom.rawLabels.map((c, i) => [String(i), c]));
            answers[qKey] = {
                type: "score",
                score: finalScore,
                confidence: Number((0.45 + (1 - scoreMetrics.variance / Math.max(1, k - 1)) * 0.45).toFixed(4)),
                legend,
                probabilities: Object.fromEntries(geom.rawLabels.map((_, i) => [String(i), Number(probs[i].toFixed(4))]))
            };
            diagnostics[qKey] = { rawSims, debiasedSims, polarAxisScore: polar.axisScore, stretched };

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

            // #29: Sentence-level extractive max-pooling for long reading comprehension passages (BoolQ)
            if (geom.domain === "boolq" && cleanState.length > 160) {
                const sentences = splitIntoSentences(cleanState);
                if (sentences.length > 1) {
                    const sentVecs = await rawEmbedFn([
                        formatQwenQuery("Find the sentence that answers the question", instText),
                        ...sentences.map(s => formatQwenQuery(instText, s))
                    ]);
                    const qVec = sentVecs[0];
                    const sVecs = sentVecs.slice(1);
                    const simsToQ = sVecs.map(sv => cosineSimilarityRaw(qVec, sv));
                    const bestIdx = simsToQ.indexOf(Math.max(...simsToQ));
                    effectiveVec = blendVectors(sVecs[bestIdx], stateVec, 0.70, 0.30);

                    // Check explicit negation mismatch between best sentence and question
                    const bestSentLower = sentences[bestIdx].toLowerCase();
                    const qHasNeg = /\b(not|never|no|cannot|can't|illegal)\b/i.test(instText);
                    const sHasNeg = /\b(not|never|cannot|can't|illegal|except|distinct\s+from|unlisted|fictional|no\s+longer|without)\b/i.test(bestSentLower);
                    if (sHasNeg && !qHasNeg) {
                        extraBoost -= 0.035;
                    } else if (!sHasNeg && !qHasNeg && simsToQ[bestIdx] > 0.68) {
                        extraBoost += 0.025;
                    }
                }
            }

            if (instText.toLowerCase().includes("ask a question") && surface.hasQuestion) {
                extraBoost += 0.12;
            }

            const simPos = cosineSimilarityRaw(effectiveVec, geom.orthoTrue);
            const simNeg = cosineSimilarityRaw(effectiveVec, geom.orthoFalse);

            // #16: Null-Input Baseline subtraction + Bayesian log-odds calibration
            const bayesian = calculateNoulBayesian(simPos, simNeg, 0.055, {
                nullPos: geom.nullPos,
                nullNeg: geom.nullNeg,
                nullWeight: 0.90,
                boost: extraBoost
            });

            answers[qKey] = {
                type: "noul",
                noul: bayesian.noul,
                probability: bayesian.probability,
                isYes: bayesian.isYes,
                isTrue: bayesian.isTrue,
                logOdds: bayesian.logOdds
            };
            diagnostics[qKey] = { simPos, simNeg, nullPos: geom.nullPos, nullNeg: geom.nullNeg, extraBoost, bayesian };
        }
    }

    return { answers, diagnostics };
}
