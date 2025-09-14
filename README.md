# 🏦 Clasificador de Ingresos con Spark ML

Este proyecto implementa un clasificador binario para predecir si una persona gana más de 50K al año usando regresión logística con Apache Spark ML.

## 📋 Descripción del Proyecto

La empresa DataPros necesita construir un modelo que permita predecir si una persona gana más de 50K al año basándose en características demográficas y laborales. El proyecto utiliza un dataset de 2000 registros simulados con las siguientes características:

- **age**: Edad de la persona (años)
- **sex**: Género (Male, Female)
- **workclass**: Tipo de empleo (Private, Self-emp, Gov)
- **fnlwgt**: Peso estadístico asociado al registro
- **education**: Nivel educativo (Bachelors, HS-grad, 11th, Masters, etc.)
- **hours_per_week**: Horas trabajadas por semana
- **label**: Clase objetivo (>50K o <=50K)

## 🏗️ Estructura del Proyecto

```
income-classifier/
├── data/
│   └── adult_income_sample.csv    # Dataset con 2000 registros
├── src/
│   ├── income_classifier.py       # Script principal de clasificación
│   └── predict_income.py          # Script para predicciones con datos nuevos
├── config/
│   └── spark_config.py            # Configuración de Spark
├── results/                        # Directorio para resultados (se crea automáticamente)
├── requirements.txt                # Dependencias de Python
└── README.md                      # Este archivo
```

## 🚀 Instalación y Configuración

### Prerrequisitos

- Python 3.8 o superior
- Java 8 o superior (requerido para Spark)
- Apache Spark 3.5.0

### Instalación

1. **Clonar o descargar el proyecto**

   ```bash
   git clone <repository-url>
   cd income-classifier
   ```

2. **Instalar dependencias**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configurar Spark (opcional)**
   - Si tienes Spark instalado localmente, asegúrate de que esté en tu PATH
   - O descarga Spark desde [https://spark.apache.org/downloads.html](https://spark.apache.org/downloads.html)

## 🎯 Uso del Proyecto

### 1. Clasificación Principal

Ejecuta el script principal para entrenar el modelo y ver los resultados:

```bash
python src/income_classifier.py
```

Este script realiza las siguientes tareas:

1. **Carga de datos**: Lee el archivo CSV y muestra estadísticas básicas
2. **Preprocesamiento**: Convierte variables categóricas usando StringIndexer y OneHotEncoder
3. **Ensamblaje de características**: Combina todas las características en un vector
4. **Entrenamiento**: Entrena un modelo de regresión logística
5. **Evaluación**: Muestra predicciones y métricas de rendimiento
6. **Predicción con datos nuevos**: Crea 9 registros nuevos y hace predicciones

### 2. Predicciones Independientes

Para realizar predicciones con datos nuevos (script de demostración):

```bash
python src/predict_income.py
```

## 🔧 Características Técnicas

### Algoritmo de Machine Learning

- **Regresión Logística** con Spark ML
- **Regularización**: Elastic Net (α=0.8, λ=0.01)
- **Máximo de iteraciones**: 100

### Preprocesamiento de Datos

- **StringIndexer**: Convierte variables categóricas a índices numéricos
- **OneHotEncoder**: Codifica variables categóricas como vectores binarios
- **VectorAssembler**: Combina todas las características en un vector

### Variables de Entrada

- **Numéricas**: age, fnlwgt, hours_per_week
- **Categóricas**: sex, workclass, education
- **Objetivo**: label (>50K o <=50K)

## 📊 Resultados Esperados

El script principal mostrará:

1. **Estadísticas del dataset**: Esquema, registros de muestra, estadísticas descriptivas
2. **Predicciones del modelo**: Resultados con probabilidades para los 2000 registros
3. **Métricas de evaluación**: AUC (Area Under Curve)
4. **Predicciones con datos nuevos**: 9 casos de ejemplo con análisis detallado

### Ejemplo de Salida

```
👤 Persona 1:
   📋 Perfil:
      • Edad: 25 años
      • Sexo: Male
      • Tipo de trabajo: Private
      • Educación: Bachelors
      • Horas por semana: 40
   🎯 Predicción: >50K
   📊 Probabilidades:
      • <=50K: 0.234 (23.4%)
      • >50K:  0.766 (76.6%)
```

## 🛠️ Personalización

### Modificar Datos de Prueba

Para cambiar los datos de prueba en `predict_income.py`, modifica la lista `sample_data` en el método `create_sample_data()`:

```python
sample_data = [
    (edad, "sexo", "tipo_trabajo", peso, "educacion", horas_semana),
    # Agregar más registros...
]
```

### Ajustar Parámetros del Modelo

En `income_classifier.py`, modifica los parámetros de la regresión logística:

```python
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label_indexed",
    maxIter=100,           # Número de iteraciones
    regParam=0.01,         # Parámetro de regularización
    elasticNetParam=0.8    # Balance entre L1 y L2
)
```

## 📈 Interpretación de Resultados

### Métricas de Rendimiento

- **AUC (Area Under Curve)**: Mide la capacidad del modelo para distinguir entre clases
  - 0.5: Rendimiento aleatorio
  - 0.7-0.8: Bueno
  - 0.8-0.9: Muy bueno
  - > 0.9: Excelente
