# Diagnóstico Profundo (480 Pruebas) y 50 Mejoras para TokenSafe Remaster

*(Copia local en el repositorio de la propuesta de 50 mejoras matemáticas, de embeddings y de código tras analizar las 480 pruebas de benchmark).*

Ver análisis detallado por bloques:
- **Bloque A (1–12):** Álgebra Lineal y Geometría de Embeddings (All-but-the-Top / resta de centroide común, proyección sobre eje polar en `Score`, ortogonalización Gram-Schmidt en `Noul`, CSLS anti-hubness, partición de polaridad para ironía).
- **Bloque B (13–26):** Estadística y Calibración en `js/primitives.js` (Z-score pre-Softmax, estiramiento de varianza para $\mathbb{E}[\text{Score}]$, calibración de prior nulo con entrada vacía para corregir `mean_p`, Sparsemax, temperatura adaptativa a $K$).
- **Bloque C (27–38):** Ingeniería de Embeddings para `Qwen3-Embedding-0.6B` (plantilla nativa `Instruct: ...\nQuery: ...` vs documentos sin prefijo, ventaneo extractivo de frases para `BoolQ`, ponderación posicional post-conector adversativo *"but/however"*, limpieza de entidades HTML y metadatos constantes como `source`).
- **Bloque D (39–50):** Rendimiento y Arquitectura (inferencia ONNX en un único lote por petición, precisión completa en `cosineSimilarity` sin `.toFixed(4)` previo al Softmax, `Float32Array`, `intraOpNumThreads: 4`).
