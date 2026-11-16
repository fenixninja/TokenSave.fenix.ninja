/**
 * model.js - Fenix Vector (Embeddings) & Dimension Reduction Integration
 *
 * Implements Block D (#39-#44):
 * - Transformers.js pipeline initialization with browser cache & WebGPU/WASM fallback
 * - Bounded LRU memory cache (max 5,000 entries) & sessionStorage persistence (#44)
 * - Direct Float32Array extraction from ONNX tensor data (#41)
 * - Full-precision 8x-unrolled cosine similarity (#42, #43)
 * - UMAP 2D projection
 */

import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2";
import { UMAP } from "https://cdn.jsdelivr.net/npm/umap-js@1.4.0/+esm";
import { cosineSimilarityRaw } from "./primitives.js";

// Serve models locally from the Python/local HTTP server (/models/)
// This enables instantaneous loading from disk with 0 internet consumption.
env.useBrowserCache = true;
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = "/models/";

if (env.backends && env.backends.onnx) {
    env.backends.onnx.logLevel = "error";
}

const CACHE_KEY = "fenix_session_cached_embeddings_v2";
const MAX_CACHE_ENTRIES = 5000;
const lruCache = new Map();
let sessionLoaded = false;

function ensureCacheLoaded() {
    if (sessionLoaded) return;
    sessionLoaded = true;
    try {
        if (typeof sessionStorage !== "undefined") {
            const raw = sessionStorage.getItem(CACHE_KEY);
            if (raw) {
                const parsed = JSON.parse(raw);
                for (const [k, v] of Object.entries(parsed)) {
                    lruCache.set(k, new Float32Array(v));
                }
            }
        }
    } catch (_) {}
}

function setLruEntry(key, vec) {
    if (lruCache.has(key)) lruCache.delete(key);
    lruCache.set(key, vec);
    if (lruCache.size > MAX_CACHE_ENTRIES) {
        const oldestKey = lruCache.keys().next().value;
        lruCache.delete(oldestKey);
    }
}

export function getEmbeddingCache() {
    ensureCacheLoaded();
    return Object.fromEntries(lruCache.entries());
}

export function saveEmbeddingCache() {
    try {
        if (typeof sessionStorage !== "undefined" && lruCache.size <= 300) {
            const serializable = {};
            for (const [k, v] of lruCache.entries()) {
                serializable[k] = Array.from(v);
            }
            sessionStorage.setItem(CACHE_KEY, JSON.stringify(serializable));
        }
    } catch (_) {}
}

let embedPipeline = null;
let modelLoadingPromise = null;

/**
 * Initializes or retrieves the singleton Transformers.js embedding pipeline.
 */
export async function getEmbeddingModel(onProgress) {
    if (embedPipeline) return embedPipeline;
    if (modelLoadingPromise) return modelLoadingPromise;

    modelLoadingPromise = (async () => {
        // 1. Check if WebGPU is truly supported AND has an active hardware adapter.
        // In browsers like Brave or Chrome with hardware acceleration disabled,
        // navigator.gpu exists but requestAdapter() returns null. Attempting WebGPU
        // would download a 541 MB file, fail, and corrupt the ONNX Runtime WASM init state.
        let hasWebGPU = false;
        if (typeof navigator !== "undefined" && navigator.gpu) {
            try {
                const adapter = await navigator.gpu.requestAdapter();
                hasWebGPU = !!adapter;
            } catch (gpuProbeErr) {
                hasWebGPU = false;
            }
        }

        if (hasWebGPU) {
            try {
                embedPipeline = await pipeline(
                    "feature-extraction",
                    "onnx-community/Qwen3-Embedding-0.6B-ONNX",
                    {
                        device: "webgpu",
                        dtype: "q4f16",
                        session_options: {
                            logSeverityLevel: 3
                        },
                        progress_callback: (info) => {
                            if (typeof onProgress === "function") onProgress(info);
                        }
                    }
                );
                return embedPipeline;
            } catch (err) {
                console.warn("WebGPU initialization failed, falling back to WASM/CPU:", err);
            }
        } else {
            console.log("No WebGPU hardware adapter available or hardware acceleration disabled. Using multithreaded WASM/CPU backend directly.");
        }

        // 2. Fallback to CPU via WebAssembly (WASM)
        // Check for Cross-Origin Isolation (required by modern browsers for SharedArrayBuffer multithreading)
        const isIsolated = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
        const targetThreads = isIsolated
            ? Math.min(4, (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 2)
            : 1;

        if (env.backends && env.backends.onnx && env.backends.onnx.wasm) {
            env.backends.onnx.wasm.numThreads = targetThreads;
        }

        if (isIsolated) {
            console.log(`[WASM] Cross-Origin Isolation detectado: Multihilo activo con ${targetThreads} nucleos CPU.`);
        } else {
            console.log("[WASM] Modo monohilo activo (crossOriginIsolated inactivo). Para activar multihilo a 4 nucleos, ejecuta: python3 server.py 3000");
        }

        try {
            embedPipeline = await pipeline(
                "feature-extraction",
                "onnx-community/Qwen3-Embedding-0.6B-ONNX",
                {
                    device: "wasm",
                    dtype: "q4",
                    session_options: {
                        logSeverityLevel: 3
                    },
                    progress_callback: (info) => {
                        if (typeof onProgress === "function") onProgress(info);
                    }
                }
            );
            return embedPipeline;
        } catch (wasmQ4Err) {
            console.error("Failed to load quantized WASM model:", wasmQ4Err);
            modelLoadingPromise = null;
            throw wasmQ4Err;
        }
    })();

    return modelLoadingPromise;
}

/**
 * #41: Extracts Float32Array slices directly from ONNX Tensor output without slow .tolist() overhead.
 */
function extractVectorsFromTensor(output, batchSize) {
    if (output && output.data && output.dims && output.dims.length >= 2) {
        const dim = output.dims[output.dims.length - 1];
        const vectors = [];
        for (let i = 0; i < batchSize; i++) {
            const start = i * dim;
            vectors.push(new Float32Array(output.data.slice(start, start + dim)));
        }
        return vectors;
    }
    return output.tolist().map(v => new Float32Array(v));
}

/**
 * #39 & #41: Computes raw embeddings for an array of pre-formatted strings in a single batch.
 */
export async function computeRawEmbeddings(texts, onStatus) {
    ensureCacheLoaded();
    const model = await getEmbeddingModel(onStatus);

    const missingIndices = [];
    const missingTexts = [];

    texts.forEach((t, idx) => {
        if (!lruCache.has(t)) {
            missingIndices.push(idx);
            missingTexts.push(t);
        }
    });

    if (missingTexts.length > 0) {
        if (typeof onStatus === "function") {
            onStatus({ status: "working", message: `Generando embeddings (${missingTexts.length} nuevos)...` });
        }
        const output = await model(missingTexts, { pooling: "mean", normalize: true });
        const vectors = extractVectorsFromTensor(output, missingTexts.length);
        missingIndices.forEach((origIdx, i) => {
            setLruEntry(texts[origIdx], vectors[i]);
        });
        saveEmbeddingCache();
    }

    return texts.map(t => lruCache.get(t));
}

/**
 * Computes embeddings for an array of texts with optional prefix.
 */
export async function computeEmbeddings(texts, prefix = "Query:", onStatus) {
    const formattedTexts = texts.map(t => (prefix ? `${prefix} ${String(t).trim()}` : String(t).trim()));
    return computeRawEmbeddings(formattedTexts, onStatus);
}

/**
 * #42 & #43: Full-precision cosine similarity without premature .toFixed(4) rounding.
 */
export function cosineSimilarity(vecA, vecB) {
    return cosineSimilarityRaw(vecA, vecB);
}

/**
 * Projects high-dimensional embeddings to 2D using UMAP.
 */
export function project2DUMAP(embeddings, options = {}) {
    if (!embeddings || embeddings.length === 0) return [];
    if (embeddings.length === 1) return [[0, 0]];

    const nNeighbors = options.nNeighbors || Math.max(1, Math.min(embeddings.length - 1, 15));
    const minDist = options.minDist !== undefined ? options.minDist : 0.1;

    const umap = new UMAP({
        nComponents: 2,
        nNeighbors,
        minDist
    });

    const plainArrays = embeddings.map(v => Array.from(v));
    const coords = umap.fit(plainArrays);
    return coords.map(c => [Number(c[0].toFixed(4)), Number(c[1].toFixed(4))]);
}
