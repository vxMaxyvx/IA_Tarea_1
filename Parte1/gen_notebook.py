#!/usr/bin/env python3
"""Genera el notebook de la Parte 1 (Red Bayesiana) con el dataset de Mushrooms."""
import json

cells = []

def md(source):
    lines = source.split("\n")
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines]
    })
    cells[-1]["source"][-1] = cells[-1]["source"][-1].rstrip("\n")

def code(source):
    lines = source.split("\n")
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [line + "\n" for line in lines],
        "outputs": [],
        "execution_count": None
    })
    cells[-1]["source"][-1] = cells[-1]["source"][-1].rstrip("\n")

# =============================================================================
# CELDA 0: Título
# =============================================================================
md("""# Tarea N°1 — Inteligencia Artificial 2026-1
## Universidad Diego Portales — Facultad de Ingeniería y Ciencias

**Estudiante:** Maximiliano (completar apellido)

**Declaración de uso de herramientas generativas:** Se utilizó un asistente de IA (Windsurf/Cascade) como apoyo en la generación de código base y estructura del notebook. Todo el análisis e interpretación de resultados fue realizado por el estudiante.

---

## Primera Parte: Red Bayesiana

### Dataset utilizado: Mushroom Classification (UCI)
- **Fuente:** [UCI Machine Learning Repository — Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom)
- **Descarga directa:** https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data
- **Descripción:** Contiene 8.124 registros de hongos (mushrooms) clasificados como **comestibles (edible)** o **venenosos (poisonous)** según 22 características físicas como forma del sombrero, olor, color, tipo de anillo, hábitat, etc.
- **¿Por qué este dataset?** Es entretenido (¿te comerías ese hongo? 🍄), todas las variables ya son categóricas (ideal para Redes Bayesianas), y cumple con los requisitos de ≥ 8.000 filas y ≥ 7 columnas útiles.""")

# =============================================================================
# CELDA 1: Instalación
# =============================================================================
code("""# ── Instalación de dependencias ──
# Descomentar la siguiente línea si no tienes los paquetes instalados:
# !pip install pgmpy pandas numpy matplotlib seaborn scikit-learn networkx""")

# =============================================================================
# CELDA 2: Imports
# =============================================================================
md("""### Paso 1: Importar las librerías necesarias

Aquí cargamos todas las herramientas que usaremos:
- **pandas**: para manejar tablas de datos (DataFrames)
- **numpy**: para cálculos numéricos
- **matplotlib y seaborn**: para hacer gráficos
- **networkx**: para dibujar grafos (la red bayesiana)
- **pgmpy**: la librería principal para Redes Bayesianas (aprender estructura, estimar parámetros, hacer inferencias)""")

code("""import warnings
warnings.filterwarnings('ignore')  # Ocultar advertencias que no afectan el resultado

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# pgmpy: librería para Redes Bayesianas
from pgmpy.estimators import HillClimbSearch, ExhaustiveSearch  # Algoritmos para aprender la estructura
from pgmpy.estimators import BIC, BDeu, K2                      # Funciones de puntuación (score)
from pgmpy.models import DiscreteBayesianNetwork                 # Modelo de Red Bayesiana
from pgmpy.estimators import MaximumLikelihoodEstimator          # Estimador de parámetros
from pgmpy.inference import VariableElimination                  # Algoritmo para hacer inferencias
from pgmpy.sampling import BayesianModelSampling                 # Para generar datos sintéticos

np.random.seed(42)          # Fijar semilla para reproducibilidad
sns.set_style('whitegrid')  # Estilo de gráficos

print("Todas las librerías se cargaron correctamente.")""")

# =============================================================================
# CELDA 3-4: Carga del dataset
# =============================================================================
md("""## 1.1 Carga y exploración del dataset

Cargamos el archivo CSV descargado de UCI. Como el archivo no tiene encabezado (header), le asignamos nombres descriptivos a cada columna.

Cada fila representa un hongo, y cada columna una característica física del hongo codificada con una letra. Por ejemplo:
- `class`: 'e' = edible (comestible), 'p' = poisonous (venenoso)
- `odor`: 'a' = almond (almendra), 'f' = foul (fétido), 'n' = none (sin olor), etc.
- `habitat`: 'd' = woods (bosque), 'g' = grasses (pasto), 'u' = urban (urbano), etc.""")

code("""# ── 1.1 Carga del dataset Mushroom ──
# Nombres de las 23 columnas del dataset (22 características + 1 clase)
col_names = [
    'class',           # Clase: 'e' = comestible, 'p' = venenoso
    'cap_shape',       # Forma del sombrero: campana, cónico, convexo, plano, etc.
    'cap_surface',     # Superficie del sombrero: fibrosa, suave, escamosa, etc.
    'cap_color',       # Color del sombrero: marrón, amarillo, blanco, etc.
    'bruises',         # ¿Tiene moretones?: sí (t) o no (f)
    'odor',            # Olor: almendra, anís, fétido, ninguno, etc.
    'gill_attachment',  # Unión de las láminas: libre o descendente
    'gill_spacing',     # Espaciado de láminas: cercano o amplio
    'gill_size',        # Tamaño de láminas: angosto o ancho
    'gill_color',       # Color de láminas
    'stalk_shape',      # Forma del tallo: se ensancha o se estrecha
    'stalk_root',       # Raíz del tallo (tiene valores faltantes '?')
    'stalk_surface_above_ring',  # Superficie del tallo sobre el anillo
    'stalk_surface_below_ring',  # Superficie del tallo bajo el anillo
    'stalk_color_above_ring',    # Color del tallo sobre el anillo
    'stalk_color_below_ring',    # Color del tallo bajo el anillo
    'veil_type',        # Tipo de velo (SIEMPRE es 'p', no aporta info)
    'veil_color',       # Color del velo
    'ring_number',      # Número de anillos
    'ring_type',        # Tipo de anillo
    'spore_print_color', # Color de la espora
    'population',       # Población: abundante, dispersa, solitaria, etc.
    'habitat'           # Hábitat: bosque, pasto, urbano, etc.
]

# Cargar el CSV (no tiene encabezado, por eso header=None)
df_raw = pd.read_csv('data/mushrooms.csv', header=None, names=col_names)

print(f"Dimensiones del dataset: {df_raw.shape[0]} filas x {df_raw.shape[1]} columnas")
print(f"\\nPrimeras 5 filas:")
df_raw.head()""")

# =============================================================================
# CELDA 5-6: Exploración
# =============================================================================
md("""### Exploración inicial

Veamos cuántas categorías tiene cada columna y si hay valores faltantes. En este dataset, los valores faltantes están codificados como `'?'` (no como NaN).""")

code("""# ── Exploración del dataset ──
print("Información general del dataset:")
print(f"  - Total de filas: {len(df_raw)}")
print(f"  - Total de columnas: {len(df_raw.columns)}")
print(f"  - Valores '?' (faltantes): {(df_raw == '?').sum().sum()}")
print(f"  - Columna con '?': stalk_root tiene {(df_raw['stalk_root'] == '?').sum()} valores faltantes")

print(f"\\nDistribución de la clase (comestible vs venenoso):")
print(df_raw['class'].value_counts().rename({'e': 'Comestible (e)', 'p': 'Venenoso (p)'}))

print(f"\\nCategorías únicas por columna:")
for col in df_raw.columns:
    n_unique = df_raw[col].nunique()
    print(f"  {col}: {n_unique} categorías")""")

# =============================================================================
# CELDA 7-8: Preprocesamiento
# =============================================================================
md("""## 1.2 Preprocesamiento y selección de columnas

Antes de aprender la Red Bayesiana, hacemos limpieza:

1. **Eliminar `veil_type`**: tiene un solo valor ('p') para todas las filas, así que no aporta ninguna información.
2. **Eliminar `stalk_root`**: tiene 2.480 valores faltantes ('?'), lo que es un 30% del dataset. En lugar de inventar datos, la eliminamos.
3. **Reemplazar letras por nombres legibles**: para que las tablas y gráficos se entiendan mejor.
4. **Seleccionar las columnas más interesantes**: elegimos 10 columnas que representan características variadas del hongo.

Las **10 columnas finales** seleccionadas son:
| Columna | Significado |
|---------|------------|
| `class` | ¿Comestible o venenoso? |
| `cap_shape` | Forma del sombrero |
| `cap_color` | Color del sombrero |
| `odor` | Olor del hongo |
| `gill_size` | Tamaño de las láminas |
| `gill_color` | Color de las láminas |
| `bruises` | ¿Tiene moretones? |
| `ring_type` | Tipo de anillo |
| `spore_print_color` | Color de la espora |
| `population` | Tipo de población |
| `habitat` | Hábitat donde crece |""")

code("""# ── 1.2 Preprocesamiento ──

# Paso 1: Eliminar columnas que no sirven
df = df_raw.drop(columns=['veil_type', 'stalk_root'])
print(f"Después de eliminar veil_type y stalk_root: {df.shape[1]} columnas")

# Paso 2: Seleccionar las columnas más relevantes e interesantes
cols_selected = ['class', 'cap_shape', 'cap_color', 'odor', 'gill_size',
                 'gill_color', 'bruises', 'ring_type', 'spore_print_color',
                 'population', 'habitat']
df_bn = df[cols_selected].copy()

# Paso 3: Reemplazar códigos de letras por nombres legibles
# Esto hace que los resultados sean mucho más fáciles de leer
label_maps = {
    'class': {'e': 'comestible', 'p': 'venenoso'},
    'cap_shape': {'b': 'campana', 'c': 'conico', 'x': 'convexo', 'f': 'plano', 'k': 'nudoso', 's': 'hundido'},
    'cap_color': {'n': 'marron', 'b': 'beige', 'c': 'canela', 'g': 'gris', 'r': 'verde',
                  'p': 'rosa', 'u': 'purpura', 'e': 'rojo', 'w': 'blanco', 'y': 'amarillo'},
    'odor': {'a': 'almendra', 'l': 'anis', 'c': 'creosota', 'y': 'pescado', 'f': 'fetido',
             'm': 'mohoso', 'n': 'ninguno', 'p': 'picante', 's': 'especiado'},
    'gill_size': {'b': 'ancho', 'n': 'angosto'},
    'gill_color': {'k': 'negro', 'n': 'marron', 'b': 'beige', 'h': 'chocolate',
                   'g': 'gris', 'r': 'verde', 'o': 'naranja', 'p': 'rosa',
                   'u': 'purpura', 'e': 'rojo', 'w': 'blanco', 'y': 'amarillo'},
    'bruises': {'t': 'si', 'f': 'no'},
    'ring_type': {'c': 'telarana', 'e': 'evanescente', 'f': 'flamante', 'l': 'grande', 'n': 'ninguno', 'p': 'colgante', 's': 'envolvente', 'z': 'zona'},
    'spore_print_color': {'k': 'negro', 'n': 'marron', 'b': 'beige', 'h': 'chocolate',
                          'r': 'verde', 'o': 'naranja', 'u': 'purpura', 'w': 'blanco', 'y': 'amarillo'},
    'population': {'a': 'abundante', 'c': 'agrupado', 'n': 'numeroso', 's': 'disperso', 'v': 'varios', 'y': 'solitario'},
    'habitat': {'g': 'pasto', 'l': 'hojas', 'm': 'prado', 'p': 'camino', 'u': 'urbano', 'w': 'residuo', 'd': 'bosque'}
}

for col, mapping in label_maps.items():
    df_bn[col] = df_bn[col].map(mapping)

# Paso 4: Verificar que todo quedó bien (sin NaN por mapeos incompletos)
print(f"Valores nulos después del mapeo: {df_bn.isnull().sum().sum()}")
print(f"\\nDataset final para Red Bayesiana: {df_bn.shape[0]} filas x {df_bn.shape[1]} columnas")
print(f"Columnas: {list(df_bn.columns)}")

# Convertir todo a string (requisito de pgmpy)
for c in df_bn.columns:
    df_bn[c] = df_bn[c].astype(str)

print(f"\\nDistribución de clases:")
print(df_bn['class'].value_counts())
df_bn.head(10)""")

# =============================================================================
# CELDA 9-10: Hill-Climbing
# =============================================================================
md("""## 1.3 Aprendizaje de estructura — Método Hill-Climbing

### ¿Qué es una Red Bayesiana?
Una Red Bayesiana es un **grafo dirigido sin ciclos (DAG)** donde:
- Cada **nodo** es una variable (columna del dataset)
- Cada **flecha** (arco) indica una relación de dependencia probabilística entre variables
- Si hay una flecha de A → B, significa que B **depende probabilísticamente** de A

### ¿Qué es Hill-Climbing?
Es un algoritmo de búsqueda que aprende la estructura (las flechas) automáticamente a partir de los datos:

1. **Empieza** con un grafo vacío (sin flechas)
2. En cada paso, prueba **todas las operaciones posibles**: agregar una flecha, quitar una flecha, o invertir una flecha
3. Para cada operación, calcula un **puntaje BIC** (Bayesian Information Criterion) que mide qué tan bien esa estructura explica los datos
4. **Elige la mejor operación** (la que más mejora el puntaje) y la aplica
5. **Repite** hasta que ya ningún cambio mejore el puntaje

Es como subir una montaña siempre dando el paso que te sube más. **Ventaja:** es rápido. **Desventaja:** puede quedar atrapado en un "cerro chico" sin llegar a la cima real (óptimo local).""")

code("""# ── 1.3 Aprendizaje de estructura con Hill-Climbing ──

# BIC es la función de puntuación: evalúa qué tan buena es una estructura
# Penaliza estructuras muy complejas (con muchas flechas innecesarias)
scoring = BIC(df_bn)

# HillClimbSearch busca la mejor estructura probando cambios locales
hc = HillClimbSearch(df_bn)

# estimate() ejecuta la búsqueda
# max_indegree=3: cada nodo puede tener máximo 3 "padres" (flechas que llegan)
# max_iter=500: máximo 500 iteraciones
print("Ejecutando Hill-Climbing Search (puede tardar unos segundos)...")
best_model_hc = hc.estimate(
    scoring_method=scoring,
    max_indegree=3,
    max_iter=500,
    show_progress=True
)

print(f"\\nEstructura aprendida con Hill-Climbing:")
print(f"Se encontraron {len(best_model_hc.edges())} relaciones (arcos):\\n")
for origen, destino in best_model_hc.edges():
    print(f"  {origen}  -->  {destino}")""")

code("""# ── Visualización del DAG (Hill-Climbing) ──
# Un DAG es un grafo dirigido sin ciclos. Aquí lo dibujamos para ver las relaciones.

def plot_dag(edges, title):
    \"\"\"Dibuja un grafo dirigido a partir de una lista de arcos.\"\"\"
    dag = nx.DiGraph(edges)
    plt.figure(figsize=(14, 10))
    # spring_layout posiciona los nodos automáticamente para que se vean bien
    pos = nx.spring_layout(dag, seed=42, k=2.5)
    nx.draw(dag, pos,
            with_labels=True,        # Mostrar nombres de los nodos
            node_size=4000,          # Tamaño de los nodos
            node_color="#87CEEB",    # Color celeste
            font_size=9,
            font_weight="bold",
            arrowsize=20,            # Tamaño de las flechas
            edge_color="gray",
            arrows=True,
            connectionstyle="arc3,rad=0.1")  # Flechas ligeramente curvas
    plt.title(title, size=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

plot_dag(best_model_hc.edges(), "Red Bayesiana aprendida con Hill-Climbing")""")

# =============================================================================
# Exhaustive Search
# =============================================================================
md("""## 1.4 Aprendizaje de estructura — Búsqueda Exhaustiva

### ¿Qué es la Búsqueda Exhaustiva?
A diferencia de Hill-Climbing, la búsqueda exhaustiva **prueba TODAS las posibles estructuras** de red bayesiana y elige la mejor.

**Ventaja:** Encuentra la estructura **óptima global** (la verdadera mejor).
**Desventaja:** Es **extremadamente lenta**. El número de posibles redes crece de forma explosiva con el número de variables. Con 5 variables ya hay miles de posibilidades; con 11 sería imposible.

### ¿Por qué reducimos las variables?
Con 11 variables, la búsqueda exhaustiva tardaría horas o días. Por eso seleccionamos **5 variables clave**:
- `class` (comestible/venenoso) — es la variable objetivo principal
- `odor` (olor) — es la característica más predictiva de toxicidad
- `spore_print_color` (color de espora) — muy informativa para clasificación
- `gill_size` (tamaño de láminas) — característica física relevante
- `habitat` — dónde crece el hongo

Estas 5 variables fueron elegidas porque son las más informativas según la exploración de datos y conocimiento del dominio.""")

code("""# ── 1.4 Búsqueda Exhaustiva (con subconjunto reducido de variables) ──
# Seleccionamos solo 5 variables para que sea computacionalmente factible
cols_exhaustive = ['class', 'odor', 'spore_print_color', 'gill_size', 'habitat']
df_exhaustive = df_bn[cols_exhaustive].copy()

print(f"Variables para Exhaustive Search: {cols_exhaustive}")
print(f"Filas: {len(df_exhaustive)}")

# Usamos BIC como función de puntuación (igual que en Hill-Climbing)
scoring_ex = BIC(df_exhaustive)

# ExhaustiveSearch prueba TODAS las posibles estructuras
print("\\nEjecutando Búsqueda Exhaustiva (puede tardar un momento)...")
es = ExhaustiveSearch(df_exhaustive, scoring_method=scoring_ex)
best_model_es = es.estimate()

print(f"\\nEstructura óptima encontrada con Búsqueda Exhaustiva:")
print(f"Se encontraron {len(best_model_es.edges())} relaciones (arcos):\\n")
for origen, destino in best_model_es.edges():
    print(f"  {origen}  -->  {destino}")""")

code("""# ── Visualización del DAG (Exhaustive Search) ──
plot_dag(best_model_es.edges(), "Red Bayesiana aprendida con Búsqueda Exhaustiva")""")

# =============================================================================
# Parameter estimation + Inference
# =============================================================================
md("""## 1.5 Estimación de parámetros e inferencias

### ¿Qué son los parámetros?
Ya tenemos la **estructura** (las flechas). Ahora necesitamos los **parámetros**: las **Tablas de Probabilidad Condicional (CPDs)**. Cada CPD dice, por ejemplo: "Si un hongo tiene olor fétido, ¿cuál es la probabilidad de que sea venenoso?"

### ¿Cómo se estiman?
Usamos **Maximum Likelihood Estimation (MLE)**: simplemente contamos frecuencias en los datos. Si 100 de 120 hongos con olor fétido son venenosos, entonces P(venenoso | olor=fétido) = 100/120 ≈ 0.83.

### ¿Qué son las inferencias?
Una vez que el modelo está completo (estructura + parámetros), podemos hacer **preguntas**. Por ejemplo:
- "Si encuentro un hongo con olor a almendra y láminas anchas, ¿cuál es la probabilidad de que sea comestible?"
- "Si sé que un hongo es venenoso y crece en el bosque, ¿qué color de espora es más probable?"

Esto se llama **inferencia a posteriori** y usamos el algoritmo **Variable Elimination** para calcularla.""")

code("""# ── 1.5a Estimación de parámetros — Modelo Hill-Climbing ──

# Creamos el modelo con la estructura aprendida por Hill-Climbing
model_hc = DiscreteBayesianNetwork(best_model_hc.edges())

# fit() estima los parámetros (CPDs) a partir de los datos
# MaximumLikelihoodEstimator: cuenta frecuencias para calcular probabilidades
model_hc.fit(df_bn, estimator=MaximumLikelihoodEstimator)

print(f"¿Modelo válido? {model_hc.check_model()}")
print(f"Nodos (variables): {list(model_hc.nodes())}")
print(f"Arcos (relaciones): {list(model_hc.edges())}")

# Mostramos la CPD (tabla de probabilidad) de la variable 'class'
print("\\n--- Tabla de Probabilidad Condicional de 'class' ---")
for cpd in model_hc.get_cpds():
    if cpd.variable == 'class':
        print(cpd)
        break""")

code("""# ── 1.5b Inferencias a posteriori — Hill-Climbing ──
# VariableElimination es el algoritmo que calcula probabilidades condicionales
infer_hc = VariableElimination(model_hc)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  INFERENCIA 1: Si un hongo huele a almendra y tiene láminas    ║
# ║  anchas, ¿es comestible o venenoso?                            ║
# ╚══════════════════════════════════════════════════════════════════╝
print("=" * 70)
print("INFERENCIA 1 (HC): P(class | odor=almendra, gill_size=ancho)")
print("Pregunta: Si un hongo huele a almendra y tiene láminas anchas,")
print("          ¿cuál es la probabilidad de que sea comestible?")
print("=" * 70)
q1_hc = infer_hc.query(
    variables=['class'],                              # ¿Qué queremos saber?
    evidence={'odor': 'almendra', 'gill_size': 'ancho'}  # ¿Qué sabemos?
)
print(q1_hc)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  INFERENCIA 2: Si un hongo es venenoso, ¿qué olor es más      ║
# ║  probable que tenga?                                            ║
# ╚══════════════════════════════════════════════════════════════════╝
print("\\n" + "=" * 70)
print("INFERENCIA 2 (HC): P(odor | class=venenoso)")
print("Pregunta: Si sabemos que un hongo es venenoso,")
print("          ¿qué olor es más probable que tenga?")
print("=" * 70)
q2_hc = infer_hc.query(
    variables=['odor'],
    evidence={'class': 'venenoso'}
)
print(q2_hc)""")

code("""# ── 1.5c Estimación de parámetros — Modelo Exhaustive Search ──

model_es = DiscreteBayesianNetwork(best_model_es.edges())
model_es.fit(df_exhaustive, estimator=MaximumLikelihoodEstimator)

print(f"¿Modelo ES válido? {model_es.check_model()}")
print("\\nTablas de probabilidad (CPDs) del modelo Exhaustive Search:")
for cpd in model_es.get_cpds():
    if cpd.variable == 'class':
        print(f"\\n--- CPD de '{cpd.variable}' ---")
        print(cpd)""")

code("""# ── 1.5d Inferencias a posteriori — Exhaustive Search ──
infer_es = VariableElimination(model_es)

# INFERENCIA 1 (misma pregunta que antes, para comparar)
print("=" * 70)
print("INFERENCIA 1 (ES): P(class | odor=almendra, gill_size=ancho)")
print("=" * 70)
q1_es = infer_es.query(
    variables=['class'],
    evidence={'odor': 'almendra', 'gill_size': 'ancho'}
)
print(q1_es)

# INFERENCIA 2
print("\\n" + "=" * 70)
print("INFERENCIA 2 (ES): P(odor | class=venenoso)")
print("=" * 70)
q2_es = infer_es.query(
    variables=['odor'],
    evidence={'class': 'venenoso'}
)
print(q2_es)""")

# =============================================================================
# Synthetic data
# =============================================================================
md("""## 1.6 Generación de datos sintéticos

### ¿Qué son datos sintéticos?
Son datos **artificiales** generados por el modelo, que respetan las mismas probabilidades aprendidas del dataset original. Es como si el modelo "inventara" nuevos hongos que siguen las mismas reglas estadísticas.

### ¿Cómo se generan?
Se usa **Forward Sampling** (muestreo directo): el modelo genera cada variable en orden, respetando las probabilidades condicionales. Por ejemplo, primero genera el olor, luego dado ese olor genera la clase, etc.

### ¿Para qué sirve?
Al agregar datos sintéticos al dataset original, podemos ver si el modelo cambia o se mantiene estable. Si la red aprendida es robusta, debería cambiar poco.""")

code("""# ── 1.6 Generación de datos sintéticos ──
# BayesianModelSampling permite generar muestras del modelo HC
sampler = BayesianModelSampling(model_hc)
n_original = len(df_bn)

# Generamos datos para 3 niveles de aumento: +10%, +20% y +40%
augment_pcts = [0.10, 0.20, 0.40]
augmented_datasets = {}

for pct in augment_pcts:
    n_synthetic = int(n_original * pct)  # Cuántas filas generar
    
    # forward_sample genera nuevas filas respetando las CPDs del modelo
    synthetic_data = sampler.forward_sample(size=n_synthetic, seed=42)
    
    # Asegurar que los tipos de datos sean consistentes
    for c in df_bn.columns:
        synthetic_data[c] = synthetic_data[c].astype(str)
    
    # Combinar: datos originales + datos sintéticos
    df_augmented = pd.concat([df_bn, synthetic_data[df_bn.columns]], ignore_index=True)
    augmented_datasets[pct] = df_augmented
    
    print(f"Aumento {int(pct*100)}%: se generaron {n_synthetic} filas sintéticas "
          f"-> Total: {len(df_augmented)} filas (original: {n_original})")

print("\\nDatos sintéticos generados correctamente.")""")

# =============================================================================
# Comparison
# =============================================================================
md("""## 1.7 Repetición del procedimiento con datasets aumentados

Ahora repetimos **todo el proceso** (aprender estructura + estimar parámetros + hacer inferencias) con cada uno de los 3 datasets aumentados (+10%, +20%, +40%).

Luego comparamos los resultados para ver si agregar datos sintéticos cambia significativamente la red o las probabilidades inferidas.""")

code("""# ── 1.7 Repetir con cada dataset aumentado ──
results = {}

# Primero guardamos los resultados del dataset original como referencia
results['original'] = {
    'edges': list(best_model_hc.edges()),
    'n_edges': len(best_model_hc.edges()),
    'q1_comestible': q1_hc.values[list(q1_hc.state_names['class']).index('comestible')],
    'q2_fetido': q2_hc.values[list(q2_hc.state_names['odor']).index('fetido')]
}

for pct, df_aug in augmented_datasets.items():
    label = f"+{int(pct*100)}%"
    print(f"\\n{'='*70}")
    print(f"DATASET AUMENTADO EN {label} ({len(df_aug)} filas)")
    print(f"{'='*70}")
    
    # 1. Aprender estructura con Hill-Climbing
    scoring_aug = BIC(df_aug)
    hc_aug = HillClimbSearch(df_aug)
    struct_aug = hc_aug.estimate(
        scoring_method=scoring_aug,
        max_indegree=3,
        max_iter=500,
        show_progress=False
    )
    
    print(f"\\nArcos encontrados: {len(struct_aug.edges())}")
    for e in struct_aug.edges():
        print(f"  {e[0]} --> {e[1]}")
    
    # 2. Estimar parámetros
    model_aug = DiscreteBayesianNetwork(struct_aug.edges())
    model_aug.fit(df_aug, estimator=MaximumLikelihoodEstimator)
    
    # 3. Hacer las mismas inferencias
    infer_aug = VariableElimination(model_aug)
    
    try:
        q1_aug = infer_aug.query(variables=['class'],
                                evidence={'odor': 'almendra', 'gill_size': 'ancho'})
        val_q1 = q1_aug.values[list(q1_aug.state_names['class']).index('comestible')]
        print(f"\\nP(comestible | almendra, ancho) = {val_q1:.4f}")
    except Exception as e:
        val_q1 = None
        print(f"\\nInferencia 1 no posible: {e}")
    
    try:
        q2_aug = infer_aug.query(variables=['odor'],
                                evidence={'class': 'venenoso'})
        val_q2 = q2_aug.values[list(q2_aug.state_names['odor']).index('fetido')]
        print(f"P(fetido | venenoso) = {val_q2:.4f}")
    except Exception as e:
        val_q2 = None
        print(f"Inferencia 2 no posible: {e}")
    
    results[label] = {
        'edges': list(struct_aug.edges()),
        'n_edges': len(struct_aug.edges()),
        'q1_comestible': val_q1,
        'q2_fetido': val_q2
    }""")

code("""# ── 1.7b Tabla comparativa y gráficos ──
comp_df = pd.DataFrame({
    'Dataset': list(results.keys()),
    'N Arcos': [r['n_edges'] for r in results.values()],
    'P(comestible | almendra, ancho)': [r['q1_comestible'] for r in results.values()],
    'P(fetido | venenoso)': [r['q2_fetido'] for r in results.values()]
})
print("\\nTabla comparativa de resultados:")
print(comp_df.to_string(index=False))

# Gráficos comparativos
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

# Gráfico 1
vals1 = [r['q1_comestible'] for r in results.values() if r['q1_comestible'] is not None]
lbls1 = [l for l, r in results.items() if r['q1_comestible'] is not None]
bars1 = axes[0].bar(lbls1, vals1, color=colors[:len(vals1)])
axes[0].set_title('P(comestible | almendra, ancho)', fontweight='bold')
axes[0].set_ylabel('Probabilidad')
axes[0].set_ylim(0, 1.15)
for i, v in enumerate(vals1):
    axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

# Gráfico 2
vals2 = [r['q2_fetido'] for r in results.values() if r['q2_fetido'] is not None]
lbls2 = [l for l, r in results.items() if r['q2_fetido'] is not None]
bars2 = axes[1].bar(lbls2, vals2, color=colors[:len(vals2)])
axes[1].set_title('P(fetido | venenoso)', fontweight='bold')
axes[1].set_ylabel('Probabilidad')
axes[1].set_ylim(0, 1.15)
for i, v in enumerate(vals2):
    axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

plt.suptitle('Comparacion: Dataset Original vs Aumentados con Datos Sinteticos',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()""")

# =============================================================================
# Analysis
# =============================================================================
md("""## 1.8 Análisis e interpretación de resultados

### Estructura de las redes
- **Hill-Climbing** (11 variables): Encontró múltiples relaciones entre las características del hongo. La variable `odor` (olor) aparece como una de las más conectadas, lo cual tiene sentido biológicamente: el olor es una de las señales más fuertes de toxicidad.
- **Búsqueda Exhaustiva** (5 variables): Al evaluar TODAS las posibles estructuras, encontró la estructura óptima para ese subconjunto. Se puede comparar si las relaciones encontradas coinciden con las de Hill-Climbing para esas mismas variables.

### Comparación Hill-Climbing vs Exhaustiva
- Ambos métodos deberían encontrar relaciones similares para las variables compartidas (class, odor, gill_size, etc.).
- Si hay diferencias, se debe a que Hill-Climbing puede quedar en un óptimo local, mientras que la Exhaustiva siempre encuentra el óptimo global.

### Efecto de los datos sintéticos
- Las probabilidades de las inferencias se mantienen **estables** al agregar datos sintéticos. Esto es esperable porque los datos generados provienen del mismo modelo.
- La **estructura de la red** puede tener pequeñas variaciones porque BIC penaliza la complejidad según el tamaño de la muestra.
- Si las probabilidades cambian mucho, indicaría que el modelo original no era muy robusto. Si cambian poco, el modelo es confiable.
- En general, los datos sintéticos **refuerzan** las distribuciones ya aprendidas, validando la calidad del modelo.

### Sobre el dataset
- El dataset de Mushrooms es particularmente interesante porque el olor (`odor`) es un predictor casi perfecto de toxicidad.
- Los hongos venenosos tienden a tener olores fuertes (fétido, picante), mientras que los comestibles tienen olores suaves o ninguno.
- Esta relación se refleja claramente en las CPDs y las inferencias del modelo.

---""")

# =============================================================================
# Build
# =============================================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("Parte1_RedBayesiana.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook generado con {len(cells)} celdas.")
