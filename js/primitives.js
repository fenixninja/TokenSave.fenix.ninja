/**
 * primitives.js - Fenix Vector Decision Primitives & Advanced Mathematical Engine (50 Improvements)
 *
 * Implements:
 * - Block A (#1-#12, #42-#43): 8x unrolled full-precision cosine similarity, All-but-the-Top centroid
 *   removal, bipolar axis projection, isotropic centering, symmetric Gram-Schmidt orthogonalization,
 *   geodesic angular similarity, Matryoshka multi-scale similarity, Pearson centered cosine,
 *   CSLS hubness / null-prior correction, question residual projection, multi-anchor prototype centroids,
 *   and head-vs-tail polarity contrast for irony detection.
 * - Block B (#13-#26, #47): Z-score pre-Softmax normalization, affine variance stretching for Score,
 *   hybrid MAP/Expected Score estimator, Null-Input Baseline calibration for Noul, K-adaptive temperature,
 *   Sparsemax simplex projection, cumulative ordinal logit, Platt calibration, epistemic vs aleatoric
 *   uncertainty decomposition, adaptive Laplace smoothing, Choquet integral in MCDA, Bayes cost-sensitive
 *   thresholding, SNR significance gate, and unified isYes/isTrue/noul interface.
 */

const EPS = 1e-9;

/**
 * #42 & #43: Full-precision 64-bit cosine similarity with 8x loop unrolling (no .toFixed(4) truncation).
 *
 * @param {ArrayLike<number>} vecA
 * @param {ArrayLike<number>} vecB
 * @returns {number}
 */
export function cosineSimilarityRaw(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
    const len = vecA.length;
    let dot = 0;
    let i = 0;
    const limit = len - (len % 8);
    for (; i < limit; i += 8) {
        dot +=
            vecA[i] * vecB[i] +
            vecA[i + 1] * vecB[i + 1] +
            vecA[i + 2] * vecB[i + 2] +
            vecA[i + 3] * vecB[i + 3] +
            vecA[i + 4] * vecB[i + 4] +
            vecA[i + 5] * vecB[i + 5] +
            vecA[i + 6] * vecB[i + 6] +
            vecA[i + 7] * vecB[i + 7];
    }
    for (; i < len; i++) {
        dot += vecA[i] * vecB[i];
    }
    return dot;
}

/**
 * Normalizes a vector to unit L2 norm in Float64/Float32.
 *
 * @param {ArrayLike<number>} vec
 * @returns {Float32Array}
 */
export function normalizeL2(vec) {
    const len = vec.length;
    const out = new Float32Array(len);
    let sumSq = 0;
    for (let i = 0; i < len; i++) {
        sumSq += vec[i] * vec[i];
    }
    const invNorm = sumSq > EPS ? 1.0 / Math.sqrt(sumSq) : 0.0;
    for (let i = 0; i < len; i++) {
        out[i] = vec[i] * invNorm;
    }
    return out;
}

/**
 * #5: Geodesic angular similarity: 1 - arccos(u . v) / pi.
 * More sensitive to small angular differences when cos(theta) is near 0.75-0.90.
 */
export function angularSimilarity(vecA, vecB) {
    const cos = Math.max(-1.0, Math.min(1.0, cosineSimilarityRaw(vecA, vecB)));
    return 1.0 - Math.acos(cos) / Math.PI;
}

/**
 * #6: Matryoshka multi-scale similarity across D=256, 512, and full 1024 dimensions.
 */
export function matryoshkaSimilarity(vecA, vecB, weights = [0.20, 0.30, 0.50]) {
    if (!vecA || !vecB || vecA.length < 512) return cosineSimilarityRaw(vecA, vecB);
    const subCos = (dim) => {
        let dot = 0, na = 0, nb = 0;
        for (let i = 0; i < dim; i++) {
            const a = vecA[i], b = vecB[i];
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        const denom = Math.sqrt(na * nb);
        return denom > EPS ? dot / denom : 0;
    };
    const s256 = subCos(256);
    const s512 = subCos(512);
    const sFull = cosineSimilarityRaw(vecA, vecB);
    return weights[0] * s256 + weights[1] * s512 + weights[2] * sFull;
}

/**
 * #7: Centered cosine similarity (Pearson correlation across embedding dimensions).
 */
export function centeredCosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
    const n = vecA.length;
    let ma = 0, mb = 0;
    for (let i = 0; i < n; i++) {
        ma += vecA[i];
        mb += vecB[i];
    }
    ma /= n;
    mb /= n;
    let cov = 0, va = 0, vb = 0;
    for (let i = 0; i < n; i++) {
        const da = vecA[i] - ma;
        const db = vecB[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    const denom = Math.sqrt(va * vb);
    return denom > EPS ? cov / denom : 0;
}

/**
 * #1: All-but-the-Top / Common-Mode Centroid Removal.
 * Subtracts the shared centroid of a candidate option set and re-normalizes each option vector,
 * isolating the unique discriminative direction of each option.
 *
 * @param {ArrayLike<number>[]} vectors
 * @param {number} [shrinkFactor=0.85]
 * @returns {Float32Array[]}
 */
export function removeCommonCentroid(vectors, shrinkFactor = 0.85) {
    if (!vectors || vectors.length <= 1) return vectors.map(v => normalizeL2(v));
    const k = vectors.length;
    const dim = vectors[0].length;
    const centroid = new Float64Array(dim);
    for (let j = 0; j < k; j++) {
        const v = vectors[j];
        for (let i = 0; i < dim; i++) {
            centroid[i] += v[i];
        }
    }
    for (let i = 0; i < dim; i++) {
        centroid[i] = (centroid[i] / k) * shrinkFactor;
    }

    return vectors.map(v => {
        const diff = new Float32Array(dim);
        for (let i = 0; i < dim; i++) {
            diff[i] = v[i] - centroid[i];
        }
        return normalizeL2(diff);
    });
}

/**
 * #2: Bipolar Axis Projection for Ordered Score Scales (0..K-1).
 * Constructs the oriented semantic axis from minimum level (0) to maximum level (K-1)
 * (and inner axis 1 to K-2 when K >= 4) and projects the state embedding onto it.
 *
 * @param {ArrayLike<number>} stateVec
 * @param {ArrayLike<number>[]} levelVecs
 * @param {ArrayLike<number>} [nullVec=null]
 * @returns {{ axisScore: number, rawProjection: number }}
 */
export function projectPolarAxis(stateVec, levelVecs, nullVec = null) {
    const k = levelVecs.length;
    if (k <= 1) return { axisScore: 0, rawProjection: 0 };
    const dim = stateVec.length;

    const buildAxis = (vLow, vHigh) => {
        const axis = new Float32Array(dim);
        for (let i = 0; i < dim; i++) {
            axis[i] = vHigh[i] - vLow[i];
        }
        return normalizeL2(axis);
    };

    const outerAxis = buildAxis(levelVecs[0], levelVecs[k - 1]);
    let proj = cosineSimilarityRaw(stateVec, outerAxis);
    if (nullVec) {
        proj -= cosineSimilarityRaw(nullVec, outerAxis);
    }

    if (k >= 4) {
        const innerAxis = buildAxis(levelVecs[1], levelVecs[k - 2]);
        let innerProj = cosineSimilarityRaw(stateVec, innerAxis);
        if (nullVec) {
            innerProj -= cosineSimilarityRaw(nullVec, innerAxis);
        }
        proj = 0.70 * proj + 0.30 * innerProj;
    }

    // Typical differential projection lies in [-0.14, +0.14]; map smoothly via tanh to [0, k-1]
    const mid = (k - 1) / 2;
    const normalized = Math.tanh(proj * 9.5);
    const axisScore = Math.max(0, Math.min(k - 1, mid + mid * normalized));
    return { axisScore, rawProjection: proj };
}

/**
 * #3: Centers a vector against a reference mean vector and re-normalizes.
 */
export function centerAndNormalizeVector(vec, meanVec, weight = 0.5) {
    if (!meanVec || meanVec.length !== vec.length) return normalizeL2(vec);
    const out = new Float32Array(vec.length);
    for (let i = 0; i < vec.length; i++) {
        out[i] = vec[i] - weight * meanVec[i];
    }
    return normalizeL2(out);
}

/**
 * #4: Symmetric Gram-Schmidt / Löwdin orthogonalization for a contrastive Noul pair (vTrue, vFalse).
 */
export function orthogonalizePair(vTrue, vFalse, strength = 0.65) {
    const overlap = cosineSimilarityRaw(vTrue, vFalse) * strength;
    const dim = vTrue.length;
    const orthoTrue = new Float32Array(dim);
    const orthoFalse = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
        orthoTrue[i] = vTrue[i] - overlap * vFalse[i];
        orthoFalse[i] = vFalse[i] - overlap * vTrue[i];
    }
    return [normalizeL2(orthoTrue), normalizeL2(orthoFalse)];
}

/**
 * #8: Cross-Domain Similarity Local Scaling (CSLS) / Hubness & Null-Prior Bias Subtraction.
 */
export function cslsSimilarity(rawSims, hubBiases = null, alpha = 0.75) {
    if (!hubBiases || hubBiases.length !== rawSims.length) return rawSims;
    return rawSims.map((s, i) => s - alpha * hubBiases[i]);
}

/**
 * #9: Question Residual Vector for Reading Comprehension (BoolQ).
 * Removes the shared question topic component so the residual vector reflects the passage's answer.
 */
export function questionResidualVector(stateVec, questionVec, beta = 0.40) {
    if (!questionVec || questionVec.length !== stateVec.length) return stateVec;
    const proj = cosineSimilarityRaw(stateVec, questionVec) * beta;
    const out = new Float32Array(stateVec.length);
    for (let i = 0; i < stateVec.length; i++) {
        out[i] = stateVec[i] - proj * questionVec[i];
    }
    return normalizeL2(out);
}

/**
 * #10: Multi-Anchor Prototype Centroid Ensembling.
 */
export function computePrototypeCentroid(vectors) {
    if (!vectors || vectors.length === 0) return new Float32Array(0);
    if (vectors.length === 1) return normalizeL2(vectors[0]);
    const dim = vectors[0].length;
    const sum = new Float32Array(dim);
    for (const v of vectors) {
        for (let i = 0; i < dim; i++) {
            sum[i] += v[i];
        }
    }
    return normalizeL2(sum);
}

/**
 * #11: Norm certainty factor from pre-normalized embedding norm.
 */
export function normCertaintyFactor(rawNorm, refNorm = 18.0) {
    if (!rawNorm || rawNorm <= 0) return 1.0;
    return Math.max(0.75, Math.min(1.25, rawNorm / refNorm));
}

/**
 * #12: Polarity Contrast Detector between first half (setup) and second half (punchline/hashtags).
 * High contrast between setup and tail indicates irony/sarcasm.
 */
export function detectPolarityContrast(vecHead, vecTail, posVec, negVec) {
    if (!vecHead || !vecTail || !posVec || !negVec) return 0;
    const axis = new Float32Array(posVec.length);
    for (let i = 0; i < posVec.length; i++) {
        axis[i] = posVec[i] - negVec[i];
    }
    const normAxis = normalizeL2(axis);
    const polHead = cosineSimilarityRaw(vecHead, normAxis);
    const polTail = cosineSimilarityRaw(vecTail, normAxis);
    const divergence = Math.abs(polHead - polTail);
    const signFlip = polHead * polTail < 0 ? 0.04 : 0.0;
    return divergence + signFlip;
}

/**
 * #13: Z-Score Normalization of similarities before Softmax.
 */
export function zScoreNormalize(sims) {
    if (!sims || sims.length <= 1) return sims || [];
    const n = sims.length;
    const mean = sims.reduce((a, b) => a + b, 0) / n;
    let varSum = 0;
    for (let i = 0; i < n; i++) {
        const d = sims[i] - mean;
        varSum += d * d;
    }
    const std = Math.sqrt(varSum / n);
    if (std < 1e-6) return sims.map(() => 0);
    return sims.map(s => (s - mean) / std);
}

/**
 * #17: K-Adaptive Temperature scaling: T(K) = baseTemp / sqrt(max(1, ln(K))).
 */
export function adaptiveTemperature(baseTemp = 0.05, numClasses = 2) {
    const scale = Math.sqrt(Math.max(0.7, Math.log(Math.max(2, numClasses))));
    return Math.max(0.005, baseTemp / scale);
}

/**
 * #18: Sparsemax projection onto the probability simplex.
 * Assigns exact 0.0 probability to low-scoring tail options so they don't drag Score to the mean.
 */
export function sparsemax(scores, temperature = 0.04) {
    if (!Array.isArray(scores) || scores.length === 0) return [];
    const n = scores.length;
    if (n === 1) return [1.0];
    const scaled = scores.map(s => s / Math.max(temperature, 0.005));
    const sorted = [...scaled].sort((a, b) => b - a);
    let cumSum = 0;
    let k = 0;
    for (let j = 0; j < n; j++) {
        cumSum += sorted[j];
        const t = (cumSum - 1) / (j + 1);
        if (sorted[j] > t) {
            k = j + 1;
        } else {
            break;
        }
    }
    const tau = (sorted.slice(0, k).reduce((a, b) => a + b, 0) - 1) / k;
    const probs = scaled.map(z => Math.max(0, z - tau));
    const sum = probs.reduce((a, b) => a + b, 0);
    return sum > EPS ? probs.map(p => p / sum) : scores.map(() => 1 / n);
}

/**
 * #19: Cumulative Ordinal Logit probabilities for ordered levels.
 */
export function ordinalLogitProbabilities(rawSims, temperature = 0.04) {
    const k = rawSims.length;
    if (k <= 1) return [1.0];
    const z = zScoreNormalize(rawSims);
    return softmax(z, temperature * 12.0);
}

/**
 * #14 & #15: Affine Variance Stretching & Hybrid MAP/Expected Score Calculator.
 * Solves the midpoint shrinkage problem where E[S] clusters around (K-1)/2.
 */
export function stretchScoreExtremes(expectedScore, maxLevel, probs, stretchGamma = 1.85) {
    if (maxLevel <= 0) return 0;
    const mid = maxLevel / 2;
    const argmaxIdx = probs.indexOf(Math.max(...probs));

    // Parabolic / local neighborhood interpolation around argmax (#15)
    let localPeak = argmaxIdx;
    if (argmaxIdx > 0 && argmaxIdx < maxLevel) {
        const pLeft = probs[argmaxIdx - 1];
        const pCenter = probs[argmaxIdx];
        const pRight = probs[argmaxIdx + 1];
        const denom = pLeft + pCenter + pRight;
        if (denom > EPS) {
            localPeak = ((argmaxIdx - 1) * pLeft + argmaxIdx * pCenter + (argmaxIdx + 1) * pRight) / denom;
        }
    } else if (argmaxIdx === 0 && maxLevel >= 1) {
        const denom = probs[0] + probs[1];
        if (denom > EPS) localPeak = probs[1] / denom;
    } else if (argmaxIdx === maxLevel && maxLevel >= 1) {
        const denom = probs[maxLevel] + probs[maxLevel - 1];
        if (denom > EPS) localPeak = maxLevel - probs[maxLevel - 1] / denom;
    }

    const stretchedMean = mid + stretchGamma * (expectedScore - mid);
    const blended = 0.55 * localPeak + 0.45 * stretchedMean;
    return Number(Math.max(0, Math.min(maxLevel, blended)).toFixed(3));
}

/**
 * #21, #22, #23, #26: Uncertainty Decomposition (Epistemic vs Aleatoric), SNR Gate & Laplace Smoothing.
 */
export function decomposeUncertainty(rawSims, probs) {
    if (!rawSims || rawSims.length === 0) {
        return { epistemic: 1, aleatoric: 1, snr: 0, isSignificant: false };
    }
    const sortedSims = [...rawSims].sort((a, b) => b - a);
    const s1 = sortedSims[0] ?? 0;
    const s2 = sortedSims[1] ?? s1;
    const rawMargin = s1 - s2;

    const mean = rawSims.reduce((a, b) => a + b, 0) / rawSims.length;
    const std = Math.sqrt(rawSims.reduce((a, b) => a + (b - mean) ** 2, 0) / rawSims.length) || 0.01;
    const snr = rawMargin / std;

    const epistemic = Number(Math.max(0, Math.min(1, 1 - Math.max(0, s1))).toFixed(4));
    const dist = analyzeChoiceDistribution(probs);
    const aleatoric = dist.normalizedEntropy;

    return {
        epistemic,
        aleatoric,
        rawMargin: Number(rawMargin.toFixed(4)),
        snr: Number(snr.toFixed(3)),
        isSignificant: snr >= 0.35
    };
}

export function laplaceSmooth(probs, maxSim, threshold = 0.35) {
    if (!probs || probs.length === 0 || maxSim >= threshold) return probs;
    const n = probs.length;
    const alpha = Math.min(0.4, (threshold - maxSim) * 2);
    return probs.map(p => (1 - alpha) * p + alpha * (1 / n));
}

/**
 * Calculates Choice confidence from a probability distribution.
 */
export function calculateChoiceConfidence(probabilities) {
    if (!Array.isArray(probabilities) || probabilities.length === 0) return 0;
    const n = probabilities.length;
    if (n === 1) return 1.0;

    const sum = probabilities.reduce((a, b) => a + b, 0);
    const normalized = sum > 1.5
        ? probabilities.map(p => p / 100)
        : (sum > 0 ? probabilities.map(p => p / sum) : probabilities.map(() => 1 / n));

    const pMax = Math.max(...normalized);
    const conf = (n * pMax - 1) / (n - 1);
    return Number(Math.max(0, Math.min(1, conf)).toFixed(4));
}

/**
 * Extended Choice analysis including Shannon entropy and margin metrics.
 */
export function analyzeChoiceDistribution(probabilities) {
    if (!Array.isArray(probabilities) || probabilities.length === 0) {
        return { confidence: 0, entropyBits: 0, normalizedEntropy: 1, marginDelta: 0, top1: 0, top2: 0 };
    }
    const n = probabilities.length;
    const sum = probabilities.reduce((a, b) => a + b, 0);
    const p = sum > 1.5 ? probabilities.map(v => v / 100) : (sum > 0 ? probabilities.map(v => v / sum) : probabilities.map(() => 1 / n));

    const sorted = [...p].sort((a, b) => b - a);
    const top1 = sorted[0] || 0;
    const top2 = sorted[1] || 0;
    const marginDelta = Number((top1 - top2).toFixed(4));

    let entropyBits = 0;
    for (let i = 0; i < n; i++) {
        if (p[i] > 1e-12) {
            entropyBits -= p[i] * Math.log2(p[i]);
        }
    }
    const maxEntropy = Math.log2(Math.max(2, n));
    const normalizedEntropy = Number((entropyBits / maxEntropy).toFixed(4));
    const confidence = calculateChoiceConfidence(probabilities);

    return {
        confidence,
        entropyBits: Number(entropyBits.toFixed(4)),
        normalizedEntropy,
        marginDelta,
        top1: Number(top1.toFixed(4)),
        top2: Number(top2.toFixed(4))
    };
}

/**
 * Calculates continuous expected score, normalized score, variance, and standard deviation
 * from probability distribution over ordered levels 0..K-1.
 */
export function calculateScoreValue(probabilities) {
    if (!Array.isArray(probabilities) || probabilities.length === 0) {
        return { expectedScore: 0, normalizedScore: 0, variance: 0, stdDev: 0, maxLevel: 0, isUnimodal: true };
    }
    const k = probabilities.length;
    const maxLevel = k - 1;
    if (maxLevel === 0) {
        return { expectedScore: 0, normalizedScore: 1.0, variance: 0, stdDev: 0, maxLevel: 0, isUnimodal: true };
    }

    const sum = probabilities.reduce((a, b) => a + b, 0);
    const p = sum > 1.5
        ? probabilities.map(v => v / 100)
        : (sum > 0 ? probabilities.map(v => v / sum) : probabilities.map(() => 1 / k));

    let expectedScore = 0;
    let secondMoment = 0;

    for (let i = 0; i < k; i++) {
        expectedScore += i * p[i];
        secondMoment += (i * i) * p[i];
    }

    const normalizedScore = expectedScore / maxLevel;
    const variance = Math.max(0, secondMoment - (expectedScore * expectedScore));
    const stdDev = Math.sqrt(variance);

    let peaks = 0;
    for (let i = 0; i < k; i++) {
        const left = i > 0 ? p[i - 1] : -Infinity;
        const right = i < k - 1 ? p[i + 1] : -Infinity;
        if (p[i] >= left && p[i] >= right && (p[i] > left || p[i] > right)) {
            peaks++;
        }
    }

    return {
        expectedScore: Number(expectedScore.toFixed(3)),
        normalizedScore: Number(normalizedScore.toFixed(4)),
        variance: Number(variance.toFixed(4)),
        stdDev: Number(stdDev.toFixed(4)),
        maxLevel,
        isUnimodal: peaks <= 1
    };
}

/**
 * Calibrates and amplifies raw similarity margins using Softmax with temperature scaling.
 */
export function softmax(similarities, temperature = 0.05) {
    if (!Array.isArray(similarities) || similarities.length === 0) return [];
    const maxVal = Math.max(...similarities);
    const temp = Math.max(temperature, 0.001);
    const exps = similarities.map(s => Math.exp((s - maxVal) / temp));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => (sumExps > 0 ? e / sumExps : 1 / similarities.length));
}

/**
 * #16, #20, #25, #47: Evaluates binary condition (Noul) with contrastive Bayesian log-odds,
 * Null-Input Baseline calibration, Platt scaling, cost-sensitive threshold, and unified properties.
 *
 * @param {number} simPos - Cosine similarity with positive proposition
 * @param {number|null} [simNeg=null] - Cosine similarity with negative proposition
 * @param {number} [temperature=0.08] - Calibrated temperature
 * @param {{ nullPos?: number, nullNeg?: number, plattAlpha?: number, plattBeta?: number, costRatioFpFn?: number, boost?: number }} [options={}]
 * @returns {{ probability: number, noul: number, isYes: boolean, isTrue: boolean, logOdds: number, marginDelta: number, threshold: number }}
 */
export function calculateNoulBayesian(simPos, simNeg = null, temperature = 0.08, options = {}) {
    let prob = 0.5;
    let logOdds = 0;
    let marginDelta = 0;

    const nullBias = (options.nullPos !== undefined && options.nullNeg !== undefined)
        ? (options.nullPos - options.nullNeg) * (options.nullWeight ?? 0.85)
        : 0;
    const plattAlpha = options.plattAlpha ?? 1.0;
    const plattBeta = options.plattBeta ?? 0.0;
    const boost = options.boost ?? 0.0;
    const costRatio = options.costRatioFpFn ?? 1.0;
    const threshold = costRatio / (1.0 + costRatio);

    if (simNeg !== null && typeof simNeg === "number") {
        marginDelta = (simPos - simNeg) - nullBias + boost;
        logOdds = plattAlpha * (marginDelta / Math.max(temperature, 0.005)) + plattBeta;
        prob = 1 / (1 + Math.exp(-logOdds));
    } else {
        const refCenter = options.nullPos !== undefined ? options.nullPos : 0.70;
        marginDelta = simPos - refCenter + boost;
        logOdds = plattAlpha * (12 * marginDelta) + plattBeta;
        prob = 1 / (1 + Math.exp(-logOdds));
    }

    prob = Math.max(0.005, Math.min(0.995, prob));
    const pFixed = Number(prob.toFixed(4));
    const decision = pFixed >= threshold;

    return {
        probability: pFixed,
        noul: pFixed,
        isYes: decision,
        isTrue: decision,
        logOdds: Number(logOdds.toFixed(3)),
        marginDelta: Number(marginDelta.toFixed(4)),
        threshold: Number(threshold.toFixed(4))
    };
}

/**
 * #24: Combines multiple normalized dimension scores using MCDA (Arithmetic, Geometric, TOPSIS & Choquet interaction).
 */
export function calculateCompositeScore(dimensionScores, weights) {
    const keys = Object.keys(dimensionScores);
    let totalWeight = 0;
    keys.forEach(k => {
        totalWeight += (weights[k] !== undefined ? weights[k] : 1);
    });

    if (totalWeight <= 0) totalWeight = 1;

    let compositeArith = 0;
    let compositeGeomLog = 0;
    let dPlusSq = 0;
    let dMinusSq = 0;

    const breakdown = keys.map(k => {
        const rawWeight = weights[k] !== undefined ? weights[k] : 1;
        const normalizedWeight = rawWeight / totalWeight;
        const score = Math.max(0, Math.min(1, dimensionScores[k] || 0));
        const contribution = score * normalizedWeight;
        compositeArith += contribution;
        compositeGeomLog += normalizedWeight * Math.log(score + 0.005);
        dPlusSq += normalizedWeight * Math.pow(score - 1.0, 2);
        dMinusSq += normalizedWeight * Math.pow(score - 0.0, 2);

        return {
            dimension: k,
            score: Number(score.toFixed(4)),
            weight: Number(normalizedWeight.toFixed(4)),
            rawWeight,
            contribution: Number(contribution.toFixed(4))
        };
    });

    const geometricScore = Math.max(0, Math.min(1, Math.exp(compositeGeomLog)));
    const dPlus = Math.sqrt(dPlusSq);
    const dMinus = Math.sqrt(dMinusSq);
    const topsisScore = (dPlus + dMinus > 0) ? (dMinus / (dPlus + dMinus)) : compositeArith;

    // #24: Choquet integral non-linear synergy adjustment
    const sortedScores = keys.map(k => Math.max(0, Math.min(1, dimensionScores[k] || 0))).sort((a, b) => a - b);
    const minScore = sortedScores[0] ?? 0;
    const maxScore = sortedScores[sortedScores.length - 1] ?? 0;
    const choquetScore = Number((0.75 * compositeArith + 0.15 * minScore + 0.10 * maxScore).toFixed(4));

    return {
        compositeScore: Number(compositeArith.toFixed(4)),
        geometricScore: Number(geometricScore.toFixed(4)),
        topsisScore: Number(topsisScore.toFixed(4)),
        choquetScore,
        percentage: `${(compositeArith * 100).toFixed(1)}%`,
        breakdown
    };
}

/**
 * Evaluates confidence against risk threshold tiers.
 */
export function evaluateConfidenceTier(confidence, thresholds = { low: 0.60, high: 0.85 }) {
    if (confidence < thresholds.low) return "LOW";
    if (confidence >= thresholds.high) return "HIGH";
    return "MEDIUM";
}
