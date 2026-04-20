# Tarea N°1 — Inteligencia Artificial 2026-1

**Universidad Diego Portales — Facultad de Ingeniería y Ciencias**

**Integrantes:** Maximiliano Oliva y Alonso Iturra

---

## Estructura del repositorio

```
IA_Tarea_1/
├── Parte1/                         # Red Bayesiana (Maximiliano)
│   ├── Parte1_RedBayesiana.ipynb   # Notebook ejecutado - Parte 1
│   ├── data/
│   │   └── mushrooms.csv           # Dataset Mushroom (UCI)
│   ├── venv/                       # Entorno virtual Python
│   ├── gen_notebook_v2.py          # Script generador del notebook
│   └── ejemplo_tarea_IA.ipynb      # Ejemplo de referencia
│
├── Parte2/                         # Modelo Oculto de Markov (Alonso)
│   ├── Parte2.ipynb                # Notebook ejecutado - Parte 2
│   ├── UCI HAR Dataset/            # Dataset HAR (UCI)
│   ├── venv/                       # Entorno virtual Python
│   └── gen_parte2.py               # Script generador del notebook
│
├── T1_IA2026-1.pdf                 # Enunciado de la tarea
├── Rúbrica_T1_IA_2026.pdf         # Rúbrica de evaluación
└── README.md                       # Este archivo
```

## Datasets utilizados

### Parte 1: Mushroom Classification
- **Fuente:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/73/mushroom)
- **Descarga directa:** https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data
- **Descripción:** 8.124 registros de hongos clasificados como comestibles o venenosos según 22 características físicas.
- 11 columnas seleccionadas para el análisis.

### Parte 2: Human Activity Recognition Using Smartphones
- **Fuente:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- **Descripción:** Registros de movimiento de sensores de smartphone asociados a 6 actividades humanas, con 30 sujetos participantes.

## Cómo ejecutar

### Parte 1
```bash
cd Parte1
python3 -m venv venv                # Solo si no existe el venv
source venv/bin/activate
pip install pgmpy pandas numpy matplotlib seaborn scikit-learn networkx ipykernel
jupyter notebook Parte1_RedBayesiana.ipynb
```

### Parte 2
```bash
cd Parte2
python3 -m venv venv                # Solo si no existe el venv
source venv/bin/activate
pip install hmmlearn pandas numpy matplotlib seaborn ipykernel
jupyter notebook Parte2.ipynb
```

## Declaración de uso de herramientas generativas

Se utilizó Claude Opus 4.6 (Windsurf/Cascade) como apoyo en la generación de código base y estructura de los notebooks. También se utilizó Gemini para entender conceptos y resolver errores. **Todo el análisis e interpretación de resultados fue realizado por los estudiantes.**
