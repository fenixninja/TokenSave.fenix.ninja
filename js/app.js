/**
 * app.js - TypeSafe AI Playground Controller & Fenix Vector Decision Engine
 *
 * Orchestrates:
 * 1. 10 Use Cases navigation in left sidebar (loads state.json & query.json per case)
 * 2. Real-time JSON syntax & schema typing verification for State & Questions
 * 3. Line gutter synchronization and error callout management
 * 4. Fenix Vector (ONNX Embeddings) & RLCD Local calibrated inference
 * 5. TypeSafe expandable response accordion viewer & raw JSON viewer
 * 6. File upload (JSON), Clear, Format, and Revert utilities
 */

import { USE_CASES, USE_CASE_CATEGORIES } from "./use-cases.js";
import {
    getEmbeddingModel,
    computeRawEmbeddings
} from "./model.js";
import { evaluateSystemOne } from "./decision-engine.js";
import {
    validateStateJson,
    validateQueryJson,
    updateEditorGutter,
    formatJson,
    renderResponseAccordion,
    insertPrimitiveSnippet,
    showToast,
    escapeHtml
} from "./ui.js";
import { loadCaseExample } from "./case-examples-loader.js";

// Global Application State
const appState = {
    activeUseCaseId: "banking_confidence",
    rawStateText: "",
    rawQuestionsText: "",
    stateErrors: [],
    questionsErrors: [],
    modelReady: false,
    modelLoading: false,
    lastResults: [],
    viewMode: "split", // 'split' | 'editor-only' | 'response-only'
    responseView: "table" // 'table' | 'raw'
};

// DOM Element References
const el = {
    sidebarList: document.getElementById("sidebar-usecases-list"),
    btnToggleSidebar: document.getElementById("btn-toggle-sidebar"),
    sidebar: document.getElementById("sidebar"),
    activeUseCaseHeaderName: document.getElementById("active-usecase-header-name"),

    // Top Bar Actions
    btnClearAll: document.getElementById("btn-clear-all"),
    btnShare: document.getElementById("btn-share"),
    btnLayoutSplit: document.getElementById("btn-layout-split"),
    btnLayoutEditorFull: document.getElementById("btn-layout-editor-full"),
    btnLayoutResponseFull: document.getElementById("btn-layout-response-full"),
    playgroundBody: document.getElementById("playground-body"),

    // State Editor
    stateEditor: document.getElementById("state-editor"),
    stateGutter: document.getElementById("state-gutter"),
    stateValidationBadge: document.getElementById("state-validation-badge"),
    stateErrorCallout: document.getElementById("state-error-callout"),
    stateErrorText: document.getElementById("state-error-text"),
    btnClearState: document.getElementById("btn-clear-state"),
    inputLoadStateFile: document.getElementById("input-load-state-file"),

    // Questions Editor
    questionsEditor: document.getElementById("questions-editor"),
    questionsGutter: document.getElementById("questions-gutter"),
    questionsValidationBadge: document.getElementById("questions-validation-badge"),
    questionsErrorCallout: document.getElementById("questions-error-callout"),
    questionsErrorText: document.getElementById("questions-error-text"),
    btnFormatQuestions: document.getElementById("btn-format-questions"),
    btnClearQuestions: document.getElementById("btn-clear-questions"),
    inputLoadQuestionsFile: document.getElementById("input-load-questions-file"),

    // Primitives Inserters
    btnInsertNoul: document.getElementById("btn-insert-noul"),
    btnInsertScore: document.getElementById("btn-insert-score"),
    btnInsertChoice: document.getElementById("btn-insert-choice"),
    btnQuickAdd: document.getElementById("btn-quick-add"),

    // Bottom Actions
    btnLoadRlcd: document.getElementById("btn-load-rlcd"),
    rlcdStatusIndicator: document.getElementById("rlcd-status-indicator"),
    rlcdBtnText: document.getElementById("rlcd-btn-text"),
    btnRevertPreset: document.getElementById("btn-revert-preset"),
    btnRunRequest: document.getElementById("btn-run-request"),

    // Response Panel
    responseTimestamp: document.getElementById("response-timestamp"),
    tabViewTable: document.getElementById("tab-view-table"),
    tabViewRaw: document.getElementById("tab-view-raw"),
    btnToggleAllAccordions: document.getElementById("btn-toggle-all-accordions"),
    responseAccordionList: document.getElementById("response-accordion-list"),
    responseRawWrapper: document.getElementById("response-raw-wrapper"),
    rawJsonCode: document.getElementById("raw-json-code"),

    // POMDP Modal
    pomdpModalOverlay: document.getElementById("pomdp-modal-overlay"),
    btnClosePomdpModal: document.getElementById("btn-close-pomdp-modal"),
    btnCancelLoadPomdp: document.getElementById("btn-cancel-load-pomdp"),
    btnConfirmLoadPomdp: document.getElementById("btn-confirm-load-pomdp"),
    modalModelProgressBar: document.getElementById("modal-model-progress-bar"),
    modalProgressFill: document.getElementById("modal-progress-fill"),
    modalProgressText: document.getElementById("modal-progress-text"),
    modalStatusDot: document.getElementById("modal-status-dot"),
    modalBtnConfirmText: document.getElementById("modal-btn-confirm-text")
};

/**
 * Initializes the application on DOM ready.
 */
document.addEventListener("DOMContentLoaded", async () => {
    buildSidebarUseCases();
    setupEventListeners();

    // Check URL hash or default to banking_confidence
    const hashCase = window.location.hash.replace("#", "");
    const initialCase = USE_CASES[hashCase] ? hashCase : "banking_confidence";
    await loadUseCase(initialCase);
});

/**
 * Builds the 10 use cases list in the left sidebar.
 */
function buildSidebarUseCases() {
    if (!el.sidebarList) return;

    const caseIds = Object.keys(USE_CASES);
    const html = caseIds.map(id => {
        const uc = USE_CASES[id];
        const cat = USE_CASE_CATEGORIES[uc.category] || { name: "General" };
        const isActive = id === appState.activeUseCaseId;
        const iconSvg = uc.icon || `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`;

        return `
            <button type="button" class="usecase-nav-item ${isActive ? 'active' : ''}" data-case-id="${id}">
                <span class="usecase-nav-icon">${iconSvg}</span>
                <div class="case-meta-wrapper">
                    <span class="case-nav-title">${escapeHtml(uc.shortName || uc.name)}</span>
                    <span class="case-nav-category">${escapeHtml(cat.name)}</span>
                </div>
            </button>
        `;
    }).join("");

    el.sidebarList.innerHTML = html;

    // Attach click listeners
    el.sidebarList.querySelectorAll(".usecase-nav-item").forEach(item => {
        item.addEventListener("click", () => {
            const caseId = item.dataset.caseId;
            if (caseId && caseId !== appState.activeUseCaseId) {
                loadUseCase(caseId);
            }
        });
    });
}

/**
 * Loads a use case by ID and populates State and Questions editors.
 */
async function loadUseCase(caseId) {
    if (!USE_CASES[caseId]) return;

    appState.activeUseCaseId = caseId;
    window.location.hash = caseId;

    const uc = USE_CASES[caseId];

    // Update active class in sidebar
    if (el.sidebarList) {
        el.sidebarList.querySelectorAll(".usecase-nav-item").forEach(item => {
            item.classList.toggle("active", item.dataset.caseId === caseId);
        });
    }

    // Update top header title
    if (el.activeUseCaseHeaderName) {
        el.activeUseCaseHeaderName.textContent = uc.name;
    }

    try {
        // Asynchronously load the first reference example (state.json and query.json)
        const exampleData = await loadCaseExample(caseId, 0);

        // Normalize State
        const statePayload = exampleData.state.state || exampleData.state;
        appState.rawStateText = JSON.stringify(statePayload, null, 2);
        el.stateEditor.value = appState.rawStateText;

        // Normalize Query
        appState.rawQuestionsText = JSON.stringify(exampleData.query, null, 2);
        el.questionsEditor.value = appState.rawQuestionsText;

        // Validate both editors
        onStateInput();
        onQuestionsInput();

        // Response panel remains blank until user explicitly clicks Run request
        resetResponseToBlank();
        showToast(`Loaded '${uc.shortName || uc.name}' (ready to run)`, "info");
    } catch (err) {
        console.warn("Error loading case example, using fallback definitions:", err);
        // Fallback to internal definitions
        const fallbackState = {
            text: uc.defaultInput || "Sample input for evaluation",
            use_case: caseId
        };
        appState.rawStateText = JSON.stringify(fallbackState, null, 2);
        el.stateEditor.value = appState.rawStateText;

        appState.rawQuestionsText = JSON.stringify(uc.questions || {}, null, 2);
        el.questionsEditor.value = appState.rawQuestionsText;

        onStateInput();
        onQuestionsInput();
        resetResponseToBlank();
    }
}

/**
 * Resets the Response panel to a blank / unexecuted state.
 */
function resetResponseToBlank() {
    appState.lastResults = [];
    renderResponseAccordion([], el.responseAccordionList);
    if (el.rawJsonCode) {
        el.rawJsonCode.textContent = "{\n  \"status\": \"unexecuted\",\n  \"message\": \"Click 'Run request ⌘↵' to run inference.\"\n}";
    }
    if (el.responseTimestamp) {
        el.responseTimestamp.textContent = "Unexecuted";
    }
}

/**
 * Validates State Editor in real-time.
 */
function onStateInput() {
    const text = el.stateEditor.value;
    const res = validateStateJson(text);
    appState.stateErrors = res.errors;

    updateEditorGutter(el.stateEditor, el.stateGutter, res.errors);

    if (res.isValid) {
        el.stateValidationBadge.className = "validation-badge valid";
        el.stateValidationBadge.textContent = "✓ Valid JSON";
        el.stateErrorCallout.style.display = "none";
    } else {
        el.stateValidationBadge.className = "validation-badge invalid";
        el.stateValidationBadge.textContent = `! ${res.errors.length} ${res.errors.length === 1 ? 'error' : 'errors'}`;
        el.stateErrorCallout.style.display = "flex";
        el.stateErrorText.textContent = res.errors.map(e => `Line ${e.line}: ${e.message}`).join(" | ");
    }
}

/**
 * Validates Questions Editor in real-time.
 */
function onQuestionsInput() {
    const text = el.questionsEditor.value;
    const res = validateQueryJson(text);
    appState.questionsErrors = res.errors;

    updateEditorGutter(el.questionsEditor, el.questionsGutter, res.errors);

    if (res.isValid) {
        el.questionsValidationBadge.className = "validation-badge valid";
        el.questionsValidationBadge.textContent = "! 0";
        el.questionsErrorCallout.style.display = "none";
    } else {
        el.questionsValidationBadge.className = "validation-badge invalid";
        el.questionsValidationBadge.textContent = `! ${res.errors.length}`;
        el.questionsErrorCallout.style.display = "flex";
        el.questionsErrorText.textContent = res.errors.map(e => `Line ${e.line}: ${e.message}`).join(" | ");
    }
}

/**
 * Modal helpers for POMDP loading confirmation.
 */
function openPomdpModal() {
    if (el.pomdpModalOverlay) {
        el.pomdpModalOverlay.style.display = "flex";
    }
}

function closePomdpModal() {
    if (el.pomdpModalOverlay) {
        el.pomdpModalOverlay.style.display = "none";
    }
}

/**
 * Executes inference on State against Questions using Fenix Vector & POMDP.
 * Strictly requires the ONNX embedding model to be loaded.
 */
async function runInference() {
    const stateVal = validateStateJson(el.stateEditor.value);
    const questionsVal = validateQueryJson(el.questionsEditor.value);

    if (!stateVal.isValid || !questionsVal.isValid) {
        showToast("Please fix syntax and schema errors before running.", "error");
        return;
    }

    // Strict model check: if POMDP is not ready, prompt user with confirmation modal
    if (!appState.modelReady) {
        openPomdpModal();
        return;
    }

    const stateObj = stateVal.data;
    const questionsObj = questionsVal.data;

    // Helper to extract clean instruction text
    const getInstructionText = (inst) => {
        if (!inst) return "";
        if (typeof inst === "string") return inst;
        if (typeof inst === "object") {
            return inst.question || inst.text || JSON.stringify(inst);
        }
        return String(inst);
    };

    // Extract text representation from state
    let inputText = "";
    const effectiveObj = stateObj.state || stateObj;
    if (typeof effectiveObj === "string") {
        inputText = effectiveObj;
    } else if (effectiveObj.resume) {
        inputText = String(effectiveObj.resume);
    } else if (effectiveObj.user_message) {
        inputText = String(effectiveObj.user_message);
    } else if (effectiveObj.text) {
        inputText = String(effectiveObj.text);
    } else if (effectiveObj.events && Array.isArray(effectiveObj.events)) {
        const eventsSummary = effectiveObj.events.map(e => {
            if (e.type === "customer_message") return `Customer: "${e.text}"`;
            if (e.type === "tool_call") {
                const argStr = e.args ? JSON.stringify(e.args) : "";
                const resStr = e.result ? JSON.stringify(e.result) : "";
                return `Action: call tool ${e.tool}(${argStr}) -> returned ${resStr}`;
            }
            return JSON.stringify(e);
        }).join("\n");
        inputText = `Agent Goal: ${effectiveObj.goal || ""}\nAgent: ${effectiveObj.agent || ""}\nTrace Events:\n${eventsSummary}\nFinal Output: ${effectiveObj.final_message || ""}`;
    } else {
        inputText = JSON.stringify(effectiveObj, null, 2);
    }

    const startTime = performance.now();
    el.btnRunRequest.classList.add("working");
    el.btnRunRequest.disabled = true;

    try {
        const results = [];

        // #46 & v4.1: Unified evaluation via evaluateSystemOne (includes batched multi-field inference,
        // calibrated Youden thresholds, 0.65 edge stretching, and enriched semantic prototypes)
        const { answers, diagnostics } = await evaluateSystemOne(stateObj, questionsObj, computeRawEmbeddings);

        for (const [qKey, qDef] of Object.entries(questionsObj)) {
            const ans = answers[qKey];
            const diag = diagnostics[qKey] || {};
            const typeNorm = String(qDef.type || "choice").toLowerCase();
            const instText = getInstructionText(qDef.instructions);

            if (typeNorm === "choice") {
                const probsObj = ans.probabilities || {};
                const optKeys = Object.keys(probsObj);
                const options = optKeys.map(k => ({
                    key: k,
                    description: (qDef.criteria && qDef.criteria[k]) || k,
                    probability: probsObj[k] || 0,
                    pct: (probsObj[k] || 0) * 100,
                    isWinner: k === ans.choice
                }));
                options.sort((a, b) => b.pct - a.pct);

                results.push({
                    key: qKey,
                    type: "choice",
                    instructions: instText,
                    winnerOption: options[0] || { key: ans.choice, pct: 100 },
                    confidence: ans.confidence || 0.8,
                    entropyBits: (diag.uncertainty && diag.uncertainty.entropyBits) || 0,
                    marginDelta: (diag.uncertainty && (1 - diag.uncertainty.epistemicUncertainty)) || 0,
                    options
                });
            } else if (typeNorm === "score") {
                const probsObj = ans.probabilities || {};
                const levelKeys = Object.keys(probsObj).sort((a, b) => Number(a) - Number(b));
                const levels = levelKeys.map(k => ({
                    level: Number(k),
                    description: (ans.legend && ans.legend[k]) || (qDef.criteria && qDef.criteria[k]) || `Level ${k}`,
                    probability: probsObj[k] || 0,
                    pct: (probsObj[k] || 0) * 100
                }));

                results.push({
                    key: qKey,
                    type: "score",
                    instructions: instText,
                    expectedScore: ans.score,
                    level: ans.level !== undefined ? ans.level : Math.round(ans.score),
                    argmaxLevel: ans.argmaxLevel,
                    maxLevel: Math.max(1, levelKeys.length - 1),
                    confidence: ans.confidence || 0.75,
                    levels
                });
            } else {
                // Noul
                const trueCriteria = (qDef.criteria && (qDef.criteria.true || qDef.criteria.True || qDef.criteria["1"])) || "Affirmative and confirmed by evidence";
                const falseCriteria = (qDef.criteria && (qDef.criteria.false || qDef.criteria.False || qDef.criteria["0"])) || "Negative or unsupported";

                results.push({
                    key: qKey,
                    type: "noul",
                    instructions: instText,
                    probabilityTrue: ans.probability,
                    isTrue: ans.isTrue,
                    isYes: ans.isYes,
                    threshold: ans.threshold,
                    logOdds: ans.logOdds,
                    trueCriteria,
                    falseCriteria
                });
            }
        }

        appState.lastResults = results;

        // Render Accordion Table
        renderResponseAccordion(results, el.responseAccordionList);

        // Render Raw JSON
        const rawOutput = {
            status: "evaluated",
            engine: "Fenix Vector / POMDP",
            evaluated_at: new Date().toISOString(),
            elapsed_ms: Math.round(performance.now() - startTime),
            results: results.reduce((acc, curr) => {
                acc[curr.key] = {
                    type: curr.type,
                    value: curr.type === "choice" ? curr.winnerOption.key : (curr.type === "score" ? curr.expectedScore : (curr.probabilityTrue >= 0.5)),
                    confidence: curr.confidence || (curr.probabilityTrue ? Math.abs(curr.probabilityTrue - 0.5) * 2 : 1)
                };
                return acc;
            }, {})
        };
        el.rawJsonCode.textContent = JSON.stringify(rawOutput, null, 2);

        // Update timestamp
        const elapsed = Math.round(performance.now() - startTime);
        el.responseTimestamp.textContent = `Ran ${elapsed}ms ago`;
    } catch (err) {
        console.error("Inference execution failed:", err);
        showToast("Inference calculation failed.", "error");
    } finally {
        el.btnRunRequest.classList.remove("working");
        el.btnRunRequest.disabled = false;
    }
}

/**
 * Initializes Fenix Vector & POMDP ONNX embeddings.
 *
 * @param {boolean} [autoRun=false] - Whether to automatically run inference after loading.
 */
async function loadRlcdLocal(autoRun = false) {
    if (appState.modelLoading || appState.modelReady) {
        if (appState.modelReady && autoRun) {
            closePomdpModal();
            await runInference();
        }
        return;
    }

    appState.modelLoading = true;
    el.rlcdStatusIndicator.className = "model-status-indicator working";
    el.rlcdBtnText.textContent = "Loading POMDP...";

    if (el.modalModelProgressBar) el.modalModelProgressBar.style.display = "flex";
    if (el.modalStatusDot) el.modalStatusDot.className = "model-status-indicator working";
    if (el.modalBtnConfirmText) el.modalBtnConfirmText.textContent = "Loading...";

    showToast("Inicializando Fenix Vector & POMDP local...", "info");

    const tStartLoad = performance.now();
    let compileTimer = null;
    let compileStart = null;
    const ESTIMATED_COMPILE_SECS = 14;

    const startCompileCountdown = () => {
        if (compileTimer) return;
        compileStart = performance.now();
        if (el.modalProgressFill) {
            el.modalProgressFill.style.width = "100%";
            el.modalProgressFill.classList.add("compiling");
        }
        const updateTick = () => {
            const elapsed = (performance.now() - compileStart) / 1000;
            const remaining = Math.max(0, Math.ceil(ESTIMATED_COMPILE_SECS - elapsed));
            if (remaining > 0) {
                if (el.modalProgressText) {
                    el.modalProgressText.textContent = `Compilando sesión ONNX en CPU (~${remaining}s restantes)...`;
                }
                el.rlcdBtnText.textContent = `Compilando (~${remaining}s)`;
                if (el.modalBtnConfirmText) el.modalBtnConfirmText.textContent = `Compilando (~${remaining}s)`;
            } else {
                const totalSec = Math.round(elapsed);
                if (el.modalProgressText) {
                    el.modalProgressText.textContent = `Finalizando inicialización ONNX (${totalSec}s)...`;
                }
                el.rlcdBtnText.textContent = "Finalizando...";
                if (el.modalBtnConfirmText) el.modalBtnConfirmText.textContent = "Finalizando...";
            }
        };
        updateTick();
        compileTimer = setInterval(updateTick, 1000);
    };

    const stopCompileCountdown = () => {
        if (compileTimer) {
            clearInterval(compileTimer);
            compileTimer = null;
        }
        if (el.modalProgressFill) {
            el.modalProgressFill.classList.remove("compiling");
        }
    };

    try {
        await getEmbeddingModel((info) => {
            if (info.status === "initiate") {
                el.rlcdBtnText.textContent = "Cargando...";
                if (el.modalProgressText) el.modalProgressText.textContent = `Descargando ${info.file || "modelo"}...`;
            } else if (info.status === "progress") {
                const pct = Math.round(info.progress ?? 0);
                if (pct < 100) {
                    el.rlcdBtnText.textContent = `Descargando (${pct}%)`;
                    if (el.modalProgressFill) el.modalProgressFill.style.width = `${pct}%`;
                    if (el.modalProgressText) el.modalProgressText.textContent = `Descargando ${info.file || "modelo"}... ${pct}%`;
                } else {
                    startCompileCountdown();
                }
            } else if (info.status === "done") {
                startCompileCountdown();
            }
        });

        stopCompileCountdown();

        const elapsedSec = ((performance.now() - tStartLoad) / 1000).toFixed(1);
        console.log(`[App] ¡Modelo cargado y compilado con exito en ${elapsedSec}s! Cerrando modal...`);

        appState.modelReady = true;
        appState.modelLoading = false;

        el.rlcdStatusIndicator.className = "model-status-indicator ready";
        el.rlcdBtnText.textContent = "POMDP (Listo)";
        if (el.modalStatusDot) el.modalStatusDot.className = "model-status-indicator ready";
        if (el.modalBtnConfirmText) el.modalBtnConfirmText.textContent = "POMDP Listo";

        showToast(`Fenix Vector & POMDP listo (${elapsedSec}s).`, "success");

        closePomdpModal();

        // Run inference automatically if requested from confirmation dialog
        if (autoRun) {
            console.log("[App] Ejecutando inferencia automatica tras carga del modelo...");
            await runInference();
        }
    } catch (err) {
        stopCompileCountdown();
        console.error("[App] Error critico al inicializar POMDP:", err);
        appState.modelLoading = false;
        el.rlcdStatusIndicator.className = "model-status-indicator";
        el.rlcdBtnText.textContent = "Cargar POMDP";
        if (el.modalStatusDot) el.modalStatusDot.className = "model-status-indicator";
        if (el.modalBtnConfirmText) el.modalBtnConfirmText.textContent = "Cargar POMDP";
        if (el.modalProgressText) el.modalProgressText.textContent = `Error: ${err.message || err}`;
        showToast("Error al inicializar el modelo POMDP.", "error");
    }
}

/**
 * Set up all DOM and keyboard event listeners.
 */
function setupEventListeners() {
    // Sidebar Collapse
    if (el.btnToggleSidebar && el.sidebar) {
        el.btnToggleSidebar.addEventListener("click", () => {
            el.sidebar.classList.toggle("collapsed");
        });
    }

    // Real-time Editor Validation & Gutter Sync
    el.stateEditor.addEventListener("input", onStateInput);
    el.stateEditor.addEventListener("scroll", () => {
        el.stateGutter.scrollTop = el.stateEditor.scrollTop;
    });

    el.questionsEditor.addEventListener("input", onQuestionsInput);
    el.questionsEditor.addEventListener("scroll", () => {
        el.questionsGutter.scrollTop = el.questionsEditor.scrollTop;
    });

    // Format Questions
    el.btnFormatQuestions.addEventListener("click", () => {
        el.questionsEditor.value = formatJson(el.questionsEditor.value);
        onQuestionsInput();
        showToast("Formatted Questions JSON (2 spaces).", "info");
    });

    // Clear Buttons
    el.btnClearState.addEventListener("click", () => {
        el.stateEditor.value = "{\n  \n}";
        onStateInput();
        showToast("State cleared.", "info");
    });

    el.btnClearQuestions.addEventListener("click", () => {
        el.questionsEditor.value = "{\n  \n}";
        onQuestionsInput();
        showToast("Questions cleared.", "info");
    });

    el.btnClearAll.addEventListener("click", () => {
        el.stateEditor.value = "{\n  \n}";
        el.questionsEditor.value = "{\n  \n}";
        onStateInput();
        onQuestionsInput();
        resetResponseToBlank();
        showToast("Playground cleared.", "info");
    });

    // Share Button
    el.btnShare.addEventListener("click", () => {
        navigator.clipboard.writeText(window.location.href);
        showToast("Playground link copied to clipboard.", "success");
    });

    // File Upload: State JSON
    el.inputLoadStateFile.addEventListener("change", (e) => {
        const file = e.target.files && e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (evt) => {
            el.stateEditor.value = formatJson(evt.target.result);
            onStateInput();
            showToast(`Loaded '${file.name}' into State.`, "success");
        };
        reader.readAsText(file);
        e.target.value = "";
    });

    // File Upload: Questions JSON
    el.inputLoadQuestionsFile.addEventListener("change", (e) => {
        const file = e.target.files && e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (evt) => {
            el.questionsEditor.value = formatJson(evt.target.result);
            onQuestionsInput();
            showToast(`Loaded '${file.name}' into Questions.`, "success");
        };
        reader.readAsText(file);
        e.target.value = "";
    });

    // Primitive Template Inserters
    el.btnInsertNoul.addEventListener("click", () => insertPrimitiveSnippet(el.questionsEditor, "noul"));
    el.btnInsertScore.addEventListener("click", () => insertPrimitiveSnippet(el.questionsEditor, "score"));
    el.btnInsertChoice.addEventListener("click", () => insertPrimitiveSnippet(el.questionsEditor, "choice"));
    el.btnQuickAdd.addEventListener("click", () => insertPrimitiveSnippet(el.questionsEditor, "choice"));

    // RLCD Local Button
    el.btnLoadRlcd.addEventListener("click", loadRlcdLocal);

    // Revert Preset Button
    el.btnRevertPreset.addEventListener("click", () => {
        loadUseCase(appState.activeUseCaseId);
        showToast("Reset to use case defaults.", "info");
    });

    // Run Request Button
    el.btnRunRequest.addEventListener("click", runInference);

    // Keyboard Shortcut (Cmd+Enter / Ctrl+Enter)
    window.addEventListener("keydown", (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
            e.preventDefault();
            runInference();
        }
    });

    // Layout View Toggles (Split, Editor Full, Response Full)
    el.btnLayoutSplit.addEventListener("click", () => setLayoutMode("split"));
    el.btnLayoutEditorFull.addEventListener("click", () => setLayoutMode("editor-only"));
    el.btnLayoutResponseFull.addEventListener("click", () => setLayoutMode("response-only"));

    // Response View Toggles (Table vs Raw JSON)
    el.tabViewTable.addEventListener("click", () => setResponseView("table"));
    el.tabViewRaw.addEventListener("click", () => setResponseView("raw"));

    // POMDP Modal Listeners
    if (el.btnClosePomdpModal) {
        el.btnClosePomdpModal.addEventListener("click", closePomdpModal);
    }
    if (el.btnCancelLoadPomdp) {
        el.btnCancelLoadPomdp.addEventListener("click", closePomdpModal);
    }
    if (el.btnConfirmLoadPomdp) {
        el.btnConfirmLoadPomdp.addEventListener("click", async () => {
            await loadRlcdLocal(true);
        });
    }
    if (el.pomdpModalOverlay) {
        el.pomdpModalOverlay.addEventListener("click", (e) => {
            if (e.target === el.pomdpModalOverlay) {
                closePomdpModal();
            }
        });
    }

    // Expand/Collapse All Accordions Toggle
    let allExpanded = false;
    el.btnToggleAllAccordions.addEventListener("click", () => {
        allExpanded = !allExpanded;
        el.responseAccordionList.querySelectorAll(".accordion-item").forEach(item => {
            item.classList.toggle("expanded", allExpanded);
        });
        el.btnToggleAllAccordions.textContent = allExpanded ? "Collapse all" : "Expand all";
    });

    // Copy Raw JSON Button
    const btnCopyRaw = document.getElementById("btn-copy-raw-json");
    if (btnCopyRaw) {
        btnCopyRaw.addEventListener("click", () => {
            navigator.clipboard.writeText(el.rawJsonCode.textContent);
            showToast("JSON copied to clipboard.", "success");
        });
    }
}

/**
 * Sets layout split mode.
 */
function setLayoutMode(mode) {
    appState.viewMode = mode;
    el.btnLayoutSplit.classList.toggle("active", mode === "split");
    el.btnLayoutEditorFull.classList.toggle("active", mode === "editor-only");
    el.btnLayoutResponseFull.classList.toggle("active", mode === "response-only");

    el.playgroundBody.className = `playground-body ${mode === "split" ? "" : mode}`;
}

/**
 * Sets response view mode (Table vs Raw JSON).
 */
function setResponseView(view) {
    appState.responseView = view;
    el.tabViewTable.classList.toggle("active", view === "table");
    el.tabViewRaw.classList.toggle("active", view === "raw");

    if (view === "table") {
        el.responseAccordionList.style.display = "flex";
        el.responseRawWrapper.style.display = "none";
    } else {
        el.responseAccordionList.style.display = "none";
        el.responseRawWrapper.style.display = "block";
    }
}
