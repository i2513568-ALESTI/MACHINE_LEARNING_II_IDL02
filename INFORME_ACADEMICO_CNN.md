# Informe académico: clasificación de radiografías de tórax mediante redes neuronales convolucionales

**Asignatura:** Machine Learning II  
**Tema:** Clasificación de imágenes con CNN (PyTorch)  
**Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kaggle)  
**Implementación:** `MACHINE_LEARNING_II_IDL02.ipynb` | **Figuras:** `outputs/`

---

## 1. Introducción

### 1.1 Contexto y motivación

La **neumonía** es una infección respiratoria frecuente que, si no se detecta a tiempo, puede tener consecuencias graves. En la práctica clínica, la **radiografía de tórax** es una de las pruebas de imagen más utilizadas. El **aprendizaje profundo** se ha explorado como apoyo al especialista, no como sustituto del criterio médico.

Este trabajo corresponde a **Machine Learning II** y aborda **clasificación de imágenes** con CNN en PyTorch, análisis de datos y evaluación con métricas adecuadas ante **desbalance de clases**.

### 1.2 Origen de los datos: Kaggle

Las imágenes provienen del conjunto [**Chest X-Ray Images (Pneumonia)**](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). Tras descargarlo, se obtiene la carpeta `chest_xray/` con subdirectorios `train/` y `test/` y clases `NORMAL` y `PNEUMONIA`, cargados con `torchvision.datasets.ImageFolder`.

En **Google Colab**, la descarga se automatiza con el token privado `MI_TOKEN` (Colab Secrets → API Tokens de Kaggle), sin exponer credenciales en el código del notebook.

### 1.3 ¿De qué trata el dataset?

| Clase | Significado |
|-------|-------------|
| **NORMAL** | Radiografía sin hallazgos compatibles con neumonía |
| **PNEUMONIA** | Radiografía con signos asociados a neumonía |

**Volúmenes (ejecución del notebook):**

| Partición | NORMAL | PNEUMONIA | Total |
|-----------|--------|-----------|-------|
| Train | 1 341 | 3 875 | 5 216 |
| Test | 234 | 390 | 624 |

Las imágenes son radiografías **pediátricas** en escala de grises (JPEG). El aprendizaje usa solo la imagen y la etiqueta binaria.

### 1.4 ¿Por qué se eligió este dataset?

1. Relevancia clínica y didáctica.  
2. Formato ideal para CNN (imágenes 2D).  
3. Tamaño adecuado para entrenar en Colab con GPU.  
4. Estructura estándar (`train` / `test` por carpetas).  
5. **Desbalance** de clases, útil para practicar F1, recall por clase y pesos en la pérdida.  
6. Benchmark público y reproducible en Kaggle.

### 1.5 Objetivos del trabajo

- Pipeline completo: EDA → preprocesamiento → CNN → validación → búsqueda de hiperparámetros → test.  
- Documentar decisiones y **interpretar** resultados en validación y test.  
- Implementación en `MACHINE_LEARNING_II_IDL02.ipynb`.

---

## 2. Objetivo del proyecto

### 2.1 Objetivo general

Diseñar, implementar, entrenar y evaluar una CNN para clasificar radiografías NORMAL / PNEUMONIA, con **random search** de hiperparámetros (un trial en esta ejecución).

### 2.2 Objetivos específicos

1. EDA y cuantificación del desbalance.  
2. Preprocesamiento (128×128, normalización, aumentación).  
3. CNN con convolución, pooling, capas densas y dropout.  
4. División train/val estratificada; `test/` solo para evaluación final.  
5. Early stopping y métricas (loss, F1).  
6. Random search documentado.  
7. Accuracy, precision, recall, F1 y matriz de confusión en test.

---

## 3. Descripción del dataset

> **Referencia:** `MACHINE_LEARNING_II_IDL02.ipynb` — celdas de descarga Kaggle, §3 EDA.

### 3.1 Estructura

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Kaggle entrega **train** y **test**. El notebook subdivide `train/` en entrenamiento interno y **validación (20 % estratificado)**. La carpeta `test/` no se usa en entrenamiento ni en la búsqueda de hiperparámetros.

### 3.2 Distribución de clases

| Partición | % PNEUMONIA | Ratio PNEUMONIA / NORMAL |
|-----------|-------------|---------------------------|
| Train | 74,3 % | 2,89 : 1 |
| Test | 62,5 % | 1,67 : 1 |

**Figura 1.** Distribución de clases — `outputs/eda_class_distribution.png`

![Distribución de clases](outputs/eda_class_distribution.png)

### 3.3 Exploración visual

La primera celda del notebook muestra ejemplos aleatorios NORMAL vs PNEUMONIA tras la descarga desde Kaggle.

**Figura 2.** Ejemplos por clase — generados en la celda inicial del notebook.

---

## 4. Preprocesamiento de datos

> **Referencia:** §4 Preprocesamiento.

| Etapa | Descripción |
|-------|-------------|
| Escala de grises | 1 canal |
| Resize | 128×128 (`IMG_SIZE = 128`) |
| Normalización | μ y σ calculados **solo en train** |
| Aumentación (train) | `RandomHorizontalFlip(p=0.5)` |

**Estadísticas de normalización (salida del notebook):**

- μ ≈ **0,482**
- σ ≈ **0,235**

`Mean/Std: [0.4823058545589447] [0.2350914180278778]`

### 4.1 Entorno de ejecución

| Parámetro | Valor (perfil GPU / Colab) |
|-----------|----------------------------|
| Dispositivo | **GPU** (`Perfil: GPU \| batch=64 \| AMP=True`) |
| Batch size | 64 |
| Mixed precision (AMP) | Sí |
| Épocas baseline | 10 |
| Random search | **1 trial**, 10 épocas |
| Workers | 2 |

---

## 5. Diseño e implementación del modelo CNN

> **Referencia:** §5–§6.

### 5.1 Arquitectura `ChestXRayCNN`

Tres bloques Conv → BatchNorm → ReLU → MaxPool; cabeza densa 256 unidades + dropout + 2 salidas.

Con `base_filters = 32` (baseline): filtros **32, 64, 128**; mapa final **16×16**.

### 5.2 Partición de datos

Desde `train/` (5 216 imágenes), validación estratificada 20 %:

| Subconjunto | Tamaño |
|-------------|--------|
| Train interno | 4 173 |
| Validación | 1 043 |
| Test | 624 |

Salida: `Train: 4173 | Val: 1043 | Test: 624`

### 5.3 Pesos de clase

`CrossEntropyLoss` con pesos inversos a la frecuencia:

| Clase | Peso |
|-------|------|
| NORMAL | 1,94 |
| PNEUMONIA | 0,67 |

`Pesos: tensor([1.9448, 0.6730])`

---

## 6. Entrenamiento del modelo

> **Referencia:** §7–§8.

- **Optimizador:** Adam (`lr=1e-3` en baseline), `weight_decay=1e-4`.  
- **Early stopping** según F1 de validación (`patience` configurable).  
- Registro de **loss** y **F1** train/val por época; gráficas en `outputs/`.

### 6.1 Baseline (10 épocas)

Modelo por defecto: `ChestXRayCNN()` — 32 filtros, kernel 3, dropout 0,5.

| Época | F1 val (referencia) |
|-------|---------------------|
| 5 | 0,9762 |
| 10 | 0,9828 |

**Mejor F1 validación (baseline): 0,9828**

**Figura 3.** Curvas baseline — `outputs/baseline_curves.png`

---

## 7. Optimización de hiperparámetros

> **Referencia:** §9 Random search.

### 7.1 Espacio de búsqueda y estrategia

Grid teórico sobre `base_filters`, `kernel_size`, `lr`, `epochs`, `dropout` (32 combinaciones). En esta ejecución se configuró **`N_RANDOM_TRIALS = 1`**: se elige **una** combinación aleatoria (semilla 42) y se entrena **10 épocas**.

### 7.2 Resultado del único trial

| Trial | Filtros | Kernel | LR | Épocas | Dropout | Mejor F1 (val) |
|-------|---------|--------|-----|--------|---------|----------------|
| **1** | 16 | 3 | 0,001 | 10 | 0,5 | **0,9808** |

Configuración del trial: `{'base_filters': 16, 'kernel_size': 3, 'lr': 0.001, 'epochs': 10, 'dropout': 0.5}`

El F1 de validación del trial (0,9808) es muy similar al baseline (0,9828); con un solo trial no se explora el grid completo, pero se cumple el flujo de búsqueda documentado.

### 7.3 Modelo final

Se reentrena con los hiperparámetros del trial 1 (`base_filters=16`, `kernel_size=3`, `dropout=0.5`, `lr=0.001`, 10 épocas). Durante el entrenamiento final, el F1 de validación alcanzó valores cercanos a **0,98** (mejor época registrada ≈ 0,9797 en época 5).

Checkpoint: `outputs/best_cnn_model.pth`

**Figura 4.** Curvas modelo final — `outputs/final_curves.png`

---

## 8. Evaluación en test

> **Referencia:** §10 Modelo final y test.

### 8.1 Métricas globales

| Métrica | Valor |
|---------|-------|
| **Accuracy** | **74,84 %** (0,7484) |
| **F1-score** (ponderado) | **70,89 %** (0,7089) |

Salida del notebook: `Test — Acc: 0.7484 | F1: 0.7089`

### 8.2 Métricas por clase

| Clase | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| NORMAL | 0,96 | 0,34 | 0,50 | 234 |
| PNEUMONIA | 0,72 | 0,99 | 0,83 | 390 |

Promedios: macro F1 ≈ 0,67; weighted F1 ≈ 0,71.

### 8.3 Interpretación

- **Alta validación (~0,98 F1)** vs **test moderado (~0,71 F1 ponderado)** indica que el modelo **no generaliza igual** en la partición Kaggle de test.
- **Recall PNEUMONIA = 0,99:** casi todas las neumonías se detectan; pocos falsos negativos.
- **Recall NORMAL = 0,34:** muchos casos normales se clasifican como neumonía (falsas alarmas).
- Coherente con el **desbalance** y el sesgo hacia la clase mayoritaria en entrenamiento.

**Figura 5.** Matriz de confusión — `outputs/confusion_matrix_test.png`

![Matriz de confusión test](outputs/confusion_matrix_test.png)

Estimación: ~80 verdaderos negativos (NORMAL correctos), ~154 falsos positivos, ~386 verdaderos positivos PNEUMONIA.

---

## 9. Resumen de métricas

| Fase | F1 (val) | Notas |
|------|----------|-------|
| Baseline (10 épocas, 32 filtros) | **0,9828** | GPU, batch 64 |
| Random search (1 trial, 16 filtros) | **0,9808** | lr=0,001, dropout=0,5 |
| Modelo final (reentrenamiento) | ~0,97–0,98 | Mejor época ≈ 0,9797 |
| **Test** | **0,7089** (ponderado) | Accuracy 74,84 % |

---

## 10. Análisis crítico

1. **Validación muy optimista:** F1 val > 0,98 con 10 épocas en GPU; test cae a ~0,71.  
2. **Un solo trial:** no permite comparar hiperparámetros; conviene aumentar `N_RANDOM_TRIALS` si hay tiempo.  
3. **Clase NORMAL en test:** recall bajo (34 %) es el principal problema clínico-metodológico.  
4. **Dataset académico:** no apto para uso clínico directo.

---

## 11. Conclusiones

1. Se implementó el pipeline completo en **`MACHINE_LEARNING_II_IDL02.ipynb`** con descarga desde Kaggle (token privado en Colab).  
2. El EDA confirmó desbalance en train (~74 % PNEUMONIA).  
3. La CNN `ChestXRayCNN` con regularización y pesos de clase alcanzó **F1 val ≈ 0,98** en baseline y trial.  
4. En **test independiente**, el rendimiento fue **moderado** (F1 ponderado **0,71**, accuracy **75 %**), con fuerte sesgo hacia PNEUMONIA.  
5. Las métricas de validación deben interpretarse junto con **recall por clase en test**.

---

## 12. Recomendaciones futuras

1. Aumentar a 5–8 trials en random search para comparar filtros y learning rate.  
2. Ajustar **umbral de decisión** o coste FN/FP según prioridad clínica.  
3. Oversampling de NORMAL o focal loss.  
4. Transfer learning (ResNet/EfficientNet).  
5. Grad-CAM para interpretabilidad.

---

## Anexos

### Anexo A — Exportación a Word

```bash
cd "c:\IC\Machine_Learning_II_2026\MACHINE_LEARNING_II_IDL02"
python scripts/build_informe_docx.py
```

### Anexo B — Archivos del proyecto

| Archivo | Descripción |
|---------|-------------|
| `MACHINE_LEARNING_II_IDL02.ipynb` | Notebook principal (Colab + CNN + random search + test) |
| `INFORME_ACADEMICO_CNN.md` | Este informe |
| `outputs/` | Figuras y `best_cnn_model.pth` |

### Anexo C — Estructura del notebook

| Sección | Contenido |
|---------|-----------|
| Celda inicial | Kaggle (`MI_TOKEN`), EDA rápido, ejemplos |
| §1–2 | Imports, configuración (perfil GPU/CPU automático) |
| §3–4 | EDA y preprocesamiento |
| §5–6 | División estratificada y `ChestXRayCNN` |
| §7–8 | Entrenamiento y baseline |
| §9 | Random search (**1 trial**) |
| §10 | Modelo final y evaluación en test |

---

*Informe alineado con la ejecución registrada en `MACHINE_LEARNING_II_IDL02.ipynb` (Colab, perfil GPU, `N_RANDOM_TRIALS = 1`).*
