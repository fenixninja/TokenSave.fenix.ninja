# 📘 Documentación Técnica y Arquitectura Integral — TokenSafe Remaster

> **Sistema:** TokenSafe Remaster (Fenix Vector & POMDP System One Decision Engine)  
> **Modelo Base:** `onnx-community/Qwen3-Embedding-0.6B-ONNX` (Cuantización `q4f16`, $D=1024$)  
> **Entorno:** 100% Unificado entre Navegador (WebGPU / WASM) y Node.js CLI  
> **Ramas de Referencia:** `main` (Fases 1–4, 50 mejoras) y `feat/remaster-v3-calibration` (v3 Calibración)

---

## 1. Evolución del Sistema y Resumen de Rendimiento

TokenSafe Remaster transforma un modelo de embeddings bi-encoder ligero ($0.6\text{B}$ parámetros) en un clasificador de decisiones de alta precisión y baja latencia para entornos de producción, eliminando la necesidad de LLMs generativos lentos y costosos para tareas de clasificación, scoring ordinal y verificación booleana.

```mermaid
flowchart LR
    A["Baseline v1 (Original)<br>49.0% Accuracy (235/480)<br>Amazon 0%, SST5 15%"] --> B["Versión v2 (50 Mejoras)<br>61.5% Accuracy (295/480)<br>+12.5 pp Global<br>Amazon 45%, SST5 33%"]
    B --> C["Versión v3 (Calibración)<br>71.3% Accuracy (342/480)<br>+22.3 pp sobre Base<br>Irony 86.7%, SMS 96.7%"]
    C --> D["Versión v4 (Plan Maestro)<br>Meta: 76%–80% Accuracy<br>Latencia -65% (Batching)<br>Umbrales Youden Óptimos"]
```

### Tabla Resumen de Evolución Empírica ($N = 480$ muestras, $n=60$ por tarea)

| Tarea | Tipo | Baseline v1 (%) | Remaster v2 (%) | Remaster v3 (%) | Ganancia Total ($\Delta$) | Métrica Clave |
|---|---|---:|---:|---:|---:|---|
| `ag_news` | Choice (4 clases) | 81.7 % | **83.3 %** | **83.3 %** | **+1.6 pp** | $F_1 = 0.819$, $\text{ECE} = 0.076$ |
| `amazon` | Score (0 a 4 estrellas) | 0.0 % | 45.0 % | **50.0 %** | **+50.0 pp** | $\text{MAE} = 0.58$, $\text{Mode-Argmax}$ |
| `boolq` | Noul (Lectura/QA) | 58.3 % | 60.0 % | **65.0 %** | **+6.7 pp** | Latencia: **8.35s $\to$ 1.25s** (6.7x) |
| `emotion` | Choice (6 emociones) | 58.3 % | **60.0 %** | **60.0 %** | **+1.7 pp** | $\text{ECE} = 0.089$, $F_1 = 0.584$ |
| `irony` | Noul (Detección Ironía) | 61.7 % | 61.7 % | **86.7 %** | **+25.0 pp** | $\text{AUROC} = 0.966$, $\text{Threshold} = 0.90$ |
| `sms_spam`| Noul (Detección Spam) | 81.7 % | 81.7 % | **96.7 %** | **+15.0 pp** | $\text{AUROC} = 0.994$, $\text{Threshold} = 0.75$ |
| `sst2` | Noul (Sentimiento Binario)| 65.0 % | 66.7 % | **83.3 %** | **+18.3 pp** | $\text{AUROC} = 0.897$, $\text{Threshold} = 0.85$ |
| `sst5` | Score (0 a 4 niveles) | 15.0 % | 33.3 % | **45.0 %** | **+30.0 pp** | $\text{Spearman} = 0.73$, $\text{Mode Blend}$ |
| **TOTAL** | **Todas las 8 Tareas** | **49.0 %** (235/480) | **61.5 %** (295/480) | **71.3 %** (342/480) | **+22.3 pp** | **Zero-shot / Sin reentrenamiento** |

---

## 2. Arquitectura Global del Sistema

El sistema opera bajo un desacoplamiento estricto en cuatro capas modulares que comparten el 100% de la lógica entre el entorno Web y el entorno CLI:

```mermaid
graph TD
    subgraph UI ["Capa de Presentación"]
        Playground["Web Playground (index.html + js/app.js)"]
        WebBench["Web Benchmark Suite (bench.html + js/bench-web.js)"]
        NodeCLI["Node.js CLI (bench/eval_accuracy.mjs + test_remaster.mjs)"]
    end

    subgraph Core ["Capa de Decisión Unificada"]
        Engine["Motor Unificado: evaluateSystemOne (js/decision-engine.js)"]
        GeomCache["Caché de Geometría de Esquemas (compiledQuestionCache)"]
    end

    subgraph Preproc ["Capa de Preprocesamiento Semántico"]
        TextPrep["js/text-preprocessor.js<br>(cleanRawText, splitAdversativeClauses, selectSalientSentences, formatQwenQuery)"]
        Prototypes["Generadores de Prototipos Léxicos<br>(buildChoiceOptionPrototypes, buildScoreLevelPrototypes, buildNoulPrototypes)"]
    end

    subgraph Math ["Capa de Primitivas Matemáticas"]
        Prim["js/primitives.js<br>• Álgebra Vectorial (Gram-Schmidt, CSLS, All-but-the-Top, Polar Axis)<br>• Calibración Estadística (Z-score, Softmax, Mode Blend, Platt, Choquet)"]
    end

    subgraph Backend ["Capa de Inferencia ONNX"]
        WebGPU["Transformers.js WebGPU / WASM (js/model.js)"]
        NodeONNX["ONNX Runtime Node.js (onnxruntime-node 4 intra-threads)"]
        LRU["LRU Vector Cache (5,000 slots)"]
    end

    UI --> Engine
    Engine --> Preproc
    Engine --> Math
    Engine --> Backend
    Backend --> LRU
```

---

## 3. Catálogo Completo de Primitivas Matemáticas (`js/primitives.js`)

### Bloque A: Álgebra Vectorial y Geometría en $\mathbb{R}^D$ ($D=1024$)

#### 1. Similitud Coseno de Alta Precisión (`cosineSimilarityRaw`)
Evita la pérdida prematura de precisión por redondeo a 4 decimales. Implementa un desenrollado de bucle manual de 8 pasos (*8x loop unrolling*) para maximizar el paralelismo a nivel de instrucción (ILP) en la CPU:
$$\text{sim}(u, v) = \frac{\sum_{i=0}^{D-1} u_i \cdot v_i}{\sqrt{\sum u_i^2} \cdot \sqrt{\sum v_i^2}}$$

#### 2. Normalización $L_2$ In-place (`normalizeL2`)
Proyecta cualquier vector sobre la hipersfera unitaria $\mathbb{S}^{D-1}$:
$$\hat{v} = \frac{v}{\|v\|_2 + \epsilon}, \quad \epsilon = 10^{-12}$$

#### 3. Similitud Angular Geodésica (`angularSimilarity`)
Calcula la métrica métrica formal sobre la hipersfera:
$$\text{dist}_{ang}(u, v) = 1 - \frac{\arccos(\text{clamp}(\text{sim}(u, v), -1, 1))}{\pi}$$

#### 4. Similitud Matryoshka Jerárquica (`matryoshkaSimilarity`)
Aprovecha el orden de dimensiones de Qwen3 para combinar resoluciones $d \in \{256, 512, 1024\}$ con pesos decrecientes:
$$S_{mat} = 0.50 \cdot \text{sim}_{1024} + 0.35 \cdot \text{sim}_{512} + 0.15 \cdot \text{sim}_{256}$$

#### 5. Remoción de Centroide Común (*All-but-the-Top*) (`removeCommonCentroid`)
Elimina la primera componente dominante del espacio de embeddings que causa que todos los textos tengan similitudes basales altas ($>0.70$):
$$c = \frac{1}{K}\sum_{i=1}^K v_i, \quad v_i' = \text{normalizeL2}(v_i - \alpha \cdot c), \quad \alpha \in [0.80, 0.90]$$

#### 6. Proyección Bipolar de Eje Polar (`projectPolarAxis`)
Resuelve el colapso de las escalas ordinales (como ratings 1–5 estrellas en Amazon) proyectando la consulta sobre el vector diferencia que une los polos extremos:
$$v_{axis} = e_{max} - e_0, \quad \text{proj} = \frac{(q - c) \cdot v_{axis}}{\|v_{axis}\|_2^2}, \quad S_{polar} = \text{clamp}\left( (K-1) \cdot \frac{\text{proj} - \text{proj}_{low}}{\text{proj}_{high} - \text{proj}_{low}}, 0, K-1 \right)$$

#### 7. Ortogonalización Simétrica de Gram-Schmidt (`orthogonalizePair`)
Para preguntas binarias (Noul Sí/No), elimina el solapamiento semántico sin favorecer a ninguno de los dos polos:
$$v_{mid} = \frac{v_+ + v_-}{2}, \quad v_+' = \text{normalizeL2}(v_+ - \beta \cdot v_{mid}), \quad v_-' = \text{normalizeL2}(v_- - \beta \cdot v_{mid})$$

#### 8. Desesgo Local CSLS (*Cross-Domain Local Scaling*) (`cslsSimilarity`)
Resta la afinidad intrínseca que tienen los candidatos hacia un texto neutro (*null-input hubness*):
$$\text{CSLS}(q, c_i) = 2 \cdot \text{sim}(q, c_i) - \lambda \cdot \text{sim}(v_{null}, c_i), \quad \lambda \approx 0.70 - 0.85$$

#### 9. Detección de Contraste de Polaridad (`detectPolarityContrast`)
Calcula el salto de sentimiento entre la primera y la segunda mitad de un texto para identificar ironía o sarcasmo:
$$\Delta_{pol} = \left( \text{sim}(v_{tail}, v_+) - \text{sim}(v_{tail}, v_-) \right) - \left( \text{sim}(v_{head}, v_+) - \text{sim}(v_{head}, v_-) \right)$$

---

### Bloque B: Calibración Estadística y Probabilística

#### 10. Estandarización Z-Score de Logits (`zScoreNormalize`)
Centra la distribución de similitudes antes del softmax para estabilizar la varianza frente a cambios de escala:
$$z_i = \frac{s_i - \mu_s}{\sigma_s + \epsilon}$$

#### 11. Temperatura Adaptativa según Cardinalidad (`adaptiveTemperature`)
Ajusta la temperatura $T$ para que el modelo no se vuelva sobreconfiado cuando hay pocas opciones ni disperso cuando hay muchas ($K$ opciones):
$$T(K) = T_0 \cdot \sqrt{\frac{\ln(K)}{\ln(2)}}, \quad T_0 \approx 0.045$$

#### 12. Softmax con Escalamiento de Margen (`softmax`)
Convierte similitudes a distribución de probabilidad en el simplex unitario $\Delta^{K-1}$:
$$p_i = \frac{\exp((s_i - \max_j s_j) / T)}{\sum_k \exp((s_k - \max_j s_j) / T)}$$

#### 13. Descomposición de Incertidumbre de Shannon (`decomposeUncertainty`)
Separa la incertidumbre aleatoria (entropía de Shannon de la distribución) de la incertidumbre epistémica (cercanía a la frontera de decisión):
$$H(p) = -\sum_{i=1}^K p_i \log_2(p_i), \quad U_{epistemic} = 1 - (p_{top1} - p_{top2})$$

#### 14. Esperanza Continua y Varianza de Score (`calculateScoreValue`)
Calcula las métricas de distribución para preguntas tipo `score`:
$$E[S] = \sum_{i=0}^{K-1} i \cdot p_i, \quad \text{Var}(S) = \sum_{i=0}^{K-1} (i - E[S])^2 \cdot p_i, \quad \sigma = \sqrt{\text{Var}(S)}$$

#### 15. Estiramiento No Lineal de Varianza de Extremos (`stretchScoreExtremes`)
Contrarresta la tendencia matemática del promedio ponderado a colapsar hacia el centro de la escala:
$$u = \frac{E[S]}{K-1} - 0.5, \quad u_{stretched} = \text{sign}(u) \cdot |2u|^\gamma \cdot 0.5, \quad S_{stretched} = (u_{stretched} + 0.5) \cdot (K-1)$$

#### 16. Score Híbrido Mode-Argmax Blending (v3)
Combina la moda discreta (el nivel con mayor probabilidad) con la media estirada continua para rescatar clasificaciones perfectas en calificaciones 1–5:
$$S_{final} = \alpha \cdot \text{argmax}_i(p_i) + (1 - \alpha) \cdot \left( 0.50 \cdot S_{stretched} + 0.50 \cdot S_{polar} \right), \quad \alpha = 0.35$$

#### 17. Inferencia Bayesiana de Log-Odds y Umbral Calibrado (`calculateNoulBayesian`)
Modela preguntas booleanas calculando el log-odds contrastivo corregido por el sesgo basal y calibrado con desplazamiento de prior:
$$\Delta_m = (s_+ - s_-) - \gamma \cdot (s_{null,+} - s_{null,-}) + \text{boost}$$
$$\text{logit} = \alpha_{platt} \cdot \left( \frac{\Delta_m}{T} \right) + \beta_{platt} - \text{priorShift}$$
$$P(\text{True}) = \frac{1}{1 + \exp(-\text{logit})}, \quad \text{Decisión} = P(\text{True}) \ge \theta_{calibrado}$$

#### 18. Integración Multicriterio de Choquet (`calculateCompositeScore`)
Calcula puntuaciones compuestas considerando interacciones no aditivas y redundancia entre dimensiones mediante índices de Shapley.

---

## 4. Preprocesamiento Semántico y de Texto (`js/text-preprocessor.js`)

### Funciones Principales

1. **`formatQwenQuery(instruction, queryText)`**:  
   Genera la estructura nativa requerida por el modelo Qwen3-Embedding:
   ```text
   Instruct: <instruction>
   Query: <queryText>
   ```
2. **`cleanRawText(text)`**:  
   Normaliza espacios en blanco, comillas tipográficas, caracteres de control y entidades HTML sin alterar acentos, puntuación ni números.
3. **`formatCleanState(state)`**:  
   Serializa estados complejos de manera determinista. Si el estado es un objeto con campos semánticos (`goal`, `agent`, `events`, `final_message`), los estructura en formato conversacional comprensible.
4. **`normalizeSmsText(text)`**:  
   Desofusca números de teléfono, códigos cortos (`txt 80488`), montos en libras/dólares (`£1000`, `1k`), URLs y reemplaza jerga común de SMS para detección robusta de spam.
5. **`extractSurfaceSignals(text)`**:  
   Detecta de forma instantánea en CPU (0 ms) signos de exclamación repetidos, mayúsculas sostenidas (*shouting*), palabras clave de urgencia (*ASAP, critical, charge*) y presencia de preguntas.
6. **`splitAdversativeClauses(text)`**:  
   Detecta conectores adversativos (*"but", "however", "although", "yet"*) y separa la oración en antecedentes y consecuente, otorgando un peso del 55% a la conclusión final.
7. **`selectSalientSentences(passage, queryText, topK = 2)` (v3)**:  
   Algoritmo de saliencia léxica rápida en CPU (0.05 ms). Calcula la intersección de tokens significativos entre la pregunta y cada oración del pasaje para seleccionar las 2 oraciones más informativas sin llamar al modelo ONNX.
8. **Generadores de Prototipos Enriquecidos**:
   - `buildChoiceOptionPrototypes`: Desambigua categorías que se solapan (ej. *business* vs *sci_tech* en AG News, o las 6 emociones de Saravia).
   - `buildScoreLevelPrototypes`: Enriquece los niveles numéricos ($0..4$) con sinónimos polares ortogonales.
   - `buildNoulPrototypes`: Configura los pares semánticos contrastivos junto a su umbral óptimo $\theta$ y su `taskPriorShift`.

---

## 5. Motor de Decisión Unificado (`js/decision-engine.js`)

La función central `evaluateSystemOne(state, questions, rawEmbedFn)` implementa el ciclo de vida de evaluación:

```mermaid
sequenceDiagram
    autonumber
    actor Client as Cliente (Web UI / Benchmark CLI)
    participant Engine as evaluateSystemOne
    participant Preproc as text-preprocessor.js
    participant Cache as compiledQuestionCache
    participant ONNX as Motor ONNX (Embeddings)
    participant Math as primitives.js

    Client->>Engine: evaluateSystemOne(state, questions)
    Engine->>Preproc: formatCleanState(state)
    
    loop Por cada pregunta (qKey, qDef)
        Engine->>Cache: Obtener geometría precompilada
        alt No está en caché
            Engine->>Preproc: buildPrototypes(qDef)
            Engine->>ONNX: rawEmbedFn([nullText, protoTexts...])
            Engine->>Math: Centroid removal / Gram-Schmidt / Orthogonalize
            Engine->>Cache: Guardar geometría compilada
        end

        Engine->>Preproc: formatQwenQuery(inst, state)
        Engine->>ONNX: rawEmbedFn([queryPrompt])

        alt type == "choice"
            Engine->>Math: CSLS(query, nullSims) + Z-score + Softmax(T_adapt)
            Engine->>Math: analyzeChoiceDistribution & decomposeUncertainty
        else type == "score"
            Engine->>Math: projectPolarAxis + stretchScoreExtremes
            Engine->>Math: Mode-Argmax Blending (alpha=0.35)
        else type == "noul"
            Engine->>Math: calculateNoulBayesian(simPos, simNeg, nullBias, threshold)
        end
    end

    Engine-->>Client: { answers, diagnostics }
```

---

## 6. Pipelines Detallados de Inferencia por Tipo

### 6.1. Pipeline Choice (Clasificación Multiclase)
*Ejemplos:* `ag_news` (4 clases), `emotion` (6 clases), enrutamiento de tickets de soporte.

1. **Compilación de Candidatos**: Se generan descripciones ricas para cada opción con `buildChoiceOptionPrototypes`.
2. **Remoción de Centroide**: Se calcula $c = \frac{1}{K}\sum v_i$ y se resta a cada vector candidato ($80\%$ de intensidad).
3. **Consulta**: Se calcula el embedding del estado con la plantilla nativa `formatQwenQuery`.
4. **Desesgo CSLS**: Se calcula la similitud con cada candidato y se descuenta el sesgo hacia la consulta neutral:
   $$s_i' = 2 \cdot \text{sim}(q, c_i) - 0.70 \cdot \text{sim}(v_{null}, c_i)$$
5. **Normalización y Softmax**: Se aplica Z-score y softmax con temperatura adaptativa $T(K) = 0.045 \cdot \sqrt{\ln K / \ln 2}$.
6. **Métricas**: Se extraen `winnerOption`, `confidence`, `entropyBits` y `marginDelta`.

---

### 6.2. Pipeline Score (Regresión Ordinal Calibrada)
*Ejemplos:* Calificación 1–5 estrellas en Amazon, severidad de bugs (0–2), urgencia (0–2).

1. **Prototipos de Nivel**: Se ordenan los niveles de $0$ a $K-1$ con anclas ortogonales (`buildScoreLevelPrototypes`).
2. **Eje Polar Bipolar**: Se establece el eje $v_{axis} = e_{max} - e_0$ y se proyecta el vector de consulta para obtener `polar.axisScore`.
3. **Distribución Softmax**: Se desesgan las similitudes por CSLS ($85\%$) y se aplica softmax ($T = 0.65$).
4. **Estiramiento de Varianza**: Se calcula la esperanza continua $E[S]$ y se estira con `stretchScoreExtremes` ($\gamma = 1.75$).
5. **Mode-Argmax Blending**:
   $$S_{final} = 0.35 \cdot \text{argmax}(p) + 0.65 \cdot \left( 0.50 \cdot S_{stretched} + 0.50 \cdot S_{polar} \right)$$
6. **Resultado**: Devuelve un `score` continuo calibrado de $0.0$ a $K-1$, `argmaxLevel`, y el desglose de probabilidades por nivel.

---

### 6.3. Pipeline Noul (Verificación Booleana de Alta Separabilidad)
*Ejemplos:* Detección de fraude, SMS spam, sarcasmo/ironía, verificación de hechos en BoolQ.

1. **Polos Sí / No**: Se construyen múltiples anclas afirmativas y negativas (`buildNoulPrototypes`).
2. **Ortogonalización**: Se extraen los centroides de ambos polos y se ortogonalizan simétricamente con Gram-Schmidt.
3. **Aceleración Saliente (BoolQ)**: Si el texto es un pasaje largo, `selectSalientSentences` extrae en CPU las 2 oraciones clave para no saturar el embedding.
4. **Contraste de Polaridad (Ironía)**: Se analiza el cambio de polaridad entre el inicio y el final de la publicación.
5. **Calibración Bayesiana**: Se calcula $\Delta_m = (s_+ - s_-) - 0.90 \cdot (s_{null,+} - s_{null,-})$.
6. **Decisión por Umbral de Youden**:
   $$P(\text{True}) = \frac{1}{1 + \exp(-\text{logit})}, \quad \text{isTrue} = P(\text{True}) \ge \theta_{task}$$
   - `irony`: $\theta = 0.90$
   - `sms_spam`: $\theta = 0.75$
   - `sst2`: $\theta = 0.85$
   - `boolq`: $\theta = 0.70$

---

## 7. Herramientas de Benchmark y Suite de Verificación

El proyecto cuenta con herramientas automatizadas para evaluación continua y control de regresiones:

### 1. `bench/eval_accuracy.mjs`
Runner principal de precisión y métricas psicométricas (AUROC, Brier, ECE, Spearman, Macro-F1).
- **Ejecutar evaluación v3 (muestra rápida):**
  ```bash
  node bench/eval_accuracy.mjs --limit 5
  ```
- **Ejecutar evaluación completa de las 480 pruebas:**
  ```bash
  node bench/eval_accuracy.mjs --limit 60
  ```
- **Comparativa automática lado a lado:**
  ```bash
  node bench/eval_accuracy.mjs compare tokensafe-remaster-v2/qwen3-embedding-0.6b tokensafe-remaster-v3/qwen3-embedding-0.6b
  ```
- **Modo diagnóstico paso a paso de predicciones fallidas:**
  ```bash
  node bench/eval_accuracy.mjs --tasks irony,sst2 --explain-failures
  ```

### 2. `bench/test_remaster.mjs`
Tests unitarios de primitivas matemáticas y pruebas de integración de API System One.
- **Ejecución básica:**
  ```bash
  node bench/test_remaster.mjs
  ```
- **Ejecución con los 150 ejemplos reales del banco de casos de uso:**
  ```bash
  node bench/test_remaster.mjs --all
  ```

### 3. `bench.html` & `js/bench-web.js`
Suite gráfica interactiva para correr las evaluaciones directamente en el navegador del cliente utilizando WebGPU o WASM, con monitorización visual en tiempo real y descarga de resultados.

---

## 8. Estructura de Directorios del Repositorio

```text
tokensafe_remaster/
├── ARQUITECTURA_TOKENSAFE_REMASTER.md   # Copia de esta documentación técnica
├── PLAN_IMPLEMENTACION_50_MEJORAS.md    # Plan de fases original
├── PROPUESTA_50_MEJORAS.md              # Especificación matemática detallada
├── index.html                           # Playground principal de inferencia
├── bench.html                           # Suite web interactiva de benchmarks
├── package.json                         # Dependencias y scripts
├── js/
│   ├── app.js                           # Controlador del Playground
│   ├── bench-web.js                     # Ejecutor web de benchmarks WebGPU/WASM
│   ├── decision-engine.js               # Motor unificado evaluateSystemOne
│   ├── model.js                         # Wrapper de Transformers.js y caché LRU
│   ├── primitives.js                    # Primitivas matemáticas y estadísticas
│   ├── text-preprocessor.js             # Preprocesador léxico y de plantillas
│   ├── ui.js                            # Renderizado del acordeón y gráficos
│   ├── use-cases.js                     # Definición de presets y categorías
│   └── case-examples-loader.js          # Cargador asíncrono de ejemplos JSON
├── bench/
│   ├── eval_accuracy.mjs                # CLI de evaluación masiva y comparativa
│   ├── calibrate.mjs                    # Calibración empírica Platt / Youden
│   ├── test_remaster.mjs                # Tests unitarios e integración
│   ├── data/                            # 8 Datasets (ag_news, amazon, boolq, emotion,
│   │   ├── ag_news.jsonl                #  irony, sms_spam, sst2, sst5)
│   │   └── ...                          
│   └── results/
│       └── accuracy.jsonl               # Registro histórico de ejecuciones
└── case-use-examples/                   # 150 ejemplos reales (state.json + query.json)
    ├── banking_confidence/              #  10 categorías x 15 presets
    ├── composite_resume/
    ├── content_moderation/
    ├── financial_crime/
    ├── insurance_claims/
    ├── legal_compliance/
    ├── llm_guardrails/
    ├── semantic_linting/
    ├── support_fanout/
    └── system_one_benchmarks/
```

---

## 9. Conclusión y Garantías del Sistema

1. **Determinismo y Fidelidad:** Toda la lógica de normalización, proyección y escalado es matemática determinista. A idénticos vectores de entrada, la respuesta y las probabilidades son exactamente idénticas en cualquier navegador o servidor.
2. **Eficiencia en Recursos:** Permite clasificar y tomar decisiones en $\approx 800 - 1400\text{ ms}$ en CPUs modestas sin GPU, reduciendo el consumo de RAM a menos de $500\text{ MB}$.
3. **Escalabilidad Zero-Shot:** No requiere fine-tuning ni reentrenamiento del modelo de lenguaje; la ganancia de más de $22$ puntos porcentuales proviene exclusivamente de la corrección geométrica del espacio latente y la calibración estadística de la decisión.

---

## 10. Arquitectura de Ejecución en Navegador: WASM Multihilo, Estabilidad y Telemetría

Para permitir una ejecución 100% local, privada y sin servidores externos en máquinas de cliente (como laptops o PCs de oficina con CPU Intel Core i5 de 4 núcleos y sin aceleración por GPU dedicada), el sistema incorpora una arquitectura de inferencia WebAssembly de alta resiliencia.

```mermaid
flowchart TD
    subgraph Browser ["Navegador Web (Brave / Chrome)"]
        UI["UI / DOM / Event Loop (60 FPS)"]
        Queue["Async Mutex Queue (inferenceQueue)"]
        Chunker["Chunker (lotes de max 4 textos)"]
        Yield["Yield (await setTimeout 0)"]
        WASM["Runtime ONNX WebAssembly (4 hilos SharedArrayBuffer)"]
        LRU["Caché LRU Local (SessionStorage)"]
    end

    subgraph Host ["Servidor Local (server.py)"]
        Headers["Cabeceras de Aislamiento:<br>COOP: same-origin<br>COEP: credentialless"]
        LocalModel["/models/ (Qwen3-Embedding-0.6B-ONNX q4f16)"]
    end

    UI -->|"Petición de Inferencia"| Queue
    Queue --> Chunker
    Chunker --> Yield
    Yield -->|"UI se mantiene fluida"| UI
    Yield --> WASM
    WASM <--> LRU
    Headers -.->|"Habilita hilos"| WASM
    LocalModel -.->|"Carga offline (~540 MB)"| WASM
```

### 10.1 Servidor Local con Aislamiento Cross-Origin (`server.py`)
Los navegadores modernos restringen el uso de `SharedArrayBuffer` por motivos de seguridad (vulnerabilidades tipo Spectre). Para desbloquear los 4 hilos en WebAssembly, `server.py` inyecta las cabeceras estándar:
- `Cross-Origin-Opener-Policy: same-origin` (COOP)
- `Cross-Origin-Embedder-Policy: credentialless` (COEP): Se utiliza `credentialless` en lugar de `require-corp` para permitir que fuentes tipográficas externas (Google Fonts) y librerías CDN (Math.js, Plotly) carguen sin bloqueos CORS.
- `Cross-Origin-Resource-Policy: cross-origin` (CORP)
- `Access-Control-Allow-Origin: *`

Adicionalmente, intercepta peticiones no esenciales como `/favicon.ico` y `/.well-known/appspecific/...` respondiendo con `204 No Content` para evitar excepciones de `FileNotFoundError` y 404 en consola.

### 10.2 Sondeo Preventivo de Adaptador WebGPU
En navegadores basados en Chromium (como Brave) con la aceleración gráfica desactivada en ajustes, la propiedad `navigator.gpu` existe pero `navigator.gpu.requestAdapter()` devuelve `null`.
Si Transformers.js intenta inicializar WebGPU directamente, la promesa rechazada queda almacenada internamente en la caché del módulo (`wasmInitPromise`), lo que corrompe de forma permanente los intentos subsiguientes de inicializar el backend CPU WebAssembly.
- **Solución:** [`js/model.js`](file:///home/fenix/Documents/tokensafe_remaster/js/model.js) evalúa primero `await navigator.gpu.requestAdapter()`. Si retorna `null`, omite WebGPU inmediatamente y arranca de forma limpia el backend WASM con 4 hilos.

### 10.3 Prevención de Bloqueos (Freezes) y Control de Concurrencia WASM
Durante la inferencia en escenarios complejos (como `support_fanout` con 6 preguntas simultáneas y transcripciones de más de 3.500 caracteres), la ejecución sin control congelaba la pestaña del navegador debido a tres factores:
1. **Colisión de Concurrencia en WebAssembly:** Múltiples preguntas evaluadas en paralelo con `Promise.all()` disparaban varias llamadas simultáneas a `session.run()` sobre la misma instancia WASM, saturando los 4 hilos de CPU en contención de mutex.
   - *Solución:* Se implementó una **Cola Asíncrona Mutex (`inferenceQueue`)** en [`js/model.js`](file:///home/fenix/Documents/tokensafe_remaster/js/model.js), garantizando que las inferencias se ejecuten de manera estrictamente secuencial y segura.
2. **Explosión Cuadrática de Atención ($O(N^2)$):** El detector de conectores adversativos (*"but", "however"*) detectaba conjunciones casuales en textos largos de soporte, triplicando los prompts (full, head, tail) para cada pregunta. 6 preguntas generaban 18 secuencias de 1.000 tokens en un solo tensor.
   - *Solución:* Se restringió la división adversativa en [`js/decision-engine.js`](file:///home/fenix/Documents/tokensafe_remaster/js/decision-engine.js) exclusivamente a textos cortos (`< 600` caracteres, como reseñas de sentimiento). Para transcripciones largas, se evalúa una sola secuencia por pregunta.
3. **Sub-batching por Chunks y Yielding al Event Loop:** En lugar de pasar decenas de textos a la vez a `model()`, se procesan en lotes de máximo 4 textos (`CHUNK_SIZE = 4`), intercalando un `await new Promise(r => setTimeout(r, 0))` entre bloques. Esto libera el hilo principal de JavaScript, permitiendo que el navegador actualice la interfaz, anime los indicadores de progreso y responda al usuario en todo momento.

### 10.4 Observabilidad y Telemetría WebAssembly en Tiempo Real
La consola de desarrollo (`F12`) proporciona visibilidad granular sobre cada pase de inferencia:
```text
▼ [WASM Assembly] Lote de 6 embeddings (0 cache, 6 inferencia CPU)
    Configuracion WASM: 4 hilos CPU | Cross-Origin: Aislado (SharedArrayBuffer activo) | Backend: ONNX Runtime WASM (SIMD)
    [Entrada 1/6] "Instruct: The customer's issue was fully resolved within the session. Query: Agent Goal: ..." (1842 chars)
    [Entrada 2/6] "Instruct: Everything the agent told the customer is consistent... Query: Agent Goal: ..." (1842 chars)
    [WASM Chunk 1/2] ✔ 4 textos en 1620 ms (405.0 ms/texto) -> Float32Array[1024]
    [WASM Chunk 2/2] ✔ 2 textos en 810 ms (405.0 ms/texto) -> Float32Array[1024]
    [WASM Assembly] Resumen final: 6 textos inferidos en 2430 ms (405.0 ms/texto)
▼ [Decision Engine v4.1] Resultados Evaluados
    ✓ [Noul] issue_resolved -> TRUE (Prob: 88.4% vs Umbral 50.0%)
    ✓ [Noul] escalation_needed -> FALSE (Prob: 12.1% vs Umbral 50.0%)
    ✓ [Choice] customer_sentiment -> Ganador: "positive" (91.2%) | Confianza: 84.5%
```

### 10.5 Unificación de Memoria SPA (Playground y Benchmarks)
Anteriormente, navegar entre `index.html` y `bench.html` requería recargar la página, forzando al navegador a descargar y compilar nuevamente el grafo ONNX de 540 MB en memoria RAM (duplicando el consumo a más de 1.1 GB y requiriendo otros 14 segundos de compilación CPU).
- Se unificó la arquitectura en una **Single Page Application (SPA)** mediante pestañas en la barra superior (`Playground` y `⚡ Benchmarks`). Ambas vistas comparten exactamente la misma instancia del modelo en memoria, reduciendo la latencia de cambio a 0 ms y estabilizando el uso de RAM.

### 10.6 TokenTrim (Estilo JEV / TokenTrim Benchmark): Poda Saliente de Contexto
En casos de uso con documentos densos (como currículums de 3.500 caracteres en `composite_resume` o historiales de transacciones), evaluar 7 preguntas concatenando el State completo generaba más de 25.000 caracteres (~6.500 tokens) enviados al runtime CPU.
Debido a que el mecanismo de auto-atención en transformadores escala de forma cuadrática ($O(N^2)$):
- **Implementación de `tokenTrimState` ([`js/text-preprocessor.js`](file:///home/fenix/Documents/tokensafe_remaster/js/text-preprocessor.js)):**
  1. Si `cleanState.length > 1000`, se activa la poda saliente por pregunta.
  2. Detecta el objetivo semántico de la instrucción (ej. años de experiencia, experiencia en LLMs, mentoría, open source).
  3. Extrae exclusivamente las oraciones e ítems directamente relevantes (< 1.000 caracteres / ~200 tokens).
  4. Reduce el volumen de tokens procesados en un **82%**, acelerando la evaluación del State entre **5x y 10x** y evitando que la CPU sature la memoria en documentos largos.

### 10.7 Solución de Geometrías Pre-compiladas (Pre-baked Schemas) y Aplicación Futura a Computer-Use

```mermaid
flowchart LR
    subgraph BuildTime ["Build Time / Offline Pre-compilation"]
        Schemas["Esquemas de Decisión / Acciones GUI"] --> Compiler["Script de Pre-baking (compile_geometries.mjs)"]
        Compiler --> GeoJSON["geometries.json (Vectores Normalizados 1024-D)"]
    end

    subgraph Runtime ["Runtime / Agente en Cliente (Browser / OS)"]
        DOMState["Estado UI / Árbol Accesibilidad (AXTree)"] --> TokenTrim["TokenTrim (< 500 chars)"]
        TokenTrim --> StateVec["1 Vector de Estado (Forward Pass único ~80 ms)"]
        GeoJSON -.->|"Carga instantánea (0 ms, 0% CPU)"| Matcher["Multiplicación Matricial / Argmax Cosine (< 2 ms)"]
        StateVec --> Matcher
        Matcher --> Decision["Acción Inmediata: Click / Type / Scroll / Wait"]
    end
```

#### 1. Concepto y Funcionamiento
Actualmente, las opciones y niveles de escala de las preguntas se compilan la primera vez que se ejecutan en el navegador (lo que en esquemas grandes de 7 preguntas con 41 opciones como `composite_resume` toma ~105 s en CPUs modestas sin caché).
Dado que las instrucciones y opciones de un caso de uso son **inmutables y conocidas a priori**, la solución óptima diseñada consiste en:
- **Pre-baking en tiempo de build:** Un script Node.js (`bench/compile_geometries.mjs`) corre el modelo una sola vez y genera un archivo estático `geometries.json` que almacena:
  - Vector nulo de referencia (`nullVec`).
  - Vectores ortogonalizados y centrados (`centeredOptVecs`, `orthoTrue`, `orthoFalse`).
  - Sesgos de hubness (`nullSims`, `nullPos`, `nullNeg`).
- **Consumo a 0 ms:** Al arrancar, el cliente carga el JSON en memoria (pocos kilobytes). La compilación de preguntas pasa de **105 segundos a 0 milisegundos**.

#### 2. Aplicación Futura: Agentes de Computer-Use (Automatización de GUI y Navegadores)
En agentes autónomos de *Computer-Use* (interacción directa con sistemas operativos y navegadores web estilo Anthropic Computer Use o agentes sobre el árbol de accesibilidad AXTree), el cuello de botella tradicional es la latencia de consultar un LLM generativo remoto (1.5 a 4 segundos por cada clic o decisión elemental).

Al integrar la arquitectura TokenSafe Remaster con **Geometrías Pre-compiladas**:
1. **Espacio de Acciones Discreto Finito:** Las acciones de una interfaz son predecibles y fijas:
   - *Intenciones del agente:* `submit_form`, `cancel_dialog`, `next_page`, `scroll_down`, `focus_input`, `dismiss_cookie_banner`, `detect_captcha_blocker`.
2. **Pre-baking de Intenciones:** Todas las posibles acciones y políticas de interfaz se pre-calculan en un diccionario de geometrías latentes una sola vez.
3. **Decisiones Sub-100 ms en Cliente:**
   - El agente captura el texto del elemento interactivo o fragmento del DOM podado mediante **TokenTrim**.
   - Genera **1 solo embedding** del fragmento UI en WebAssembly (~60–80 ms).
   - Compara mediante producto punto contra las acciones pre-compiladas (< 2 ms).
   - El agente ejecuta la acción en la interfaz con **latencia casi imperceptible para el usuario humano**, de forma 100% privada, local y con 0 coste de tokens por llamada API.
