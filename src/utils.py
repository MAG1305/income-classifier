"""
Utilidades para el proyecto de clasificación de ingresos
======================================================

Funciones auxiliares para análisis de datos y visualización
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, when, isnan
import pandas as pd

def analyze_data_distribution(df: DataFrame, spark):
    """
    Analiza la distribución de los datos
    
    Args:
        df: DataFrame de Spark
        spark: Sesión de Spark
    """
    print("\n📊 ANÁLISIS DE DISTRIBUCIÓN DE DATOS")
    print("=" * 50)
    
    # Convertir a Pandas para análisis más fácil
    pandas_df = df.toPandas()
    
    # Análisis de la variable objetivo
    print("\n🎯 Distribución de la variable objetivo (label):")
    label_counts = pandas_df['label'].value_counts()
    print(label_counts)
    print(f"Proporción >50K: {label_counts.get('>50K', 0) / len(pandas_df) * 100:.1f}%")
    print(f"Proporción <=50K: {label_counts.get('<=50K', 0) / len(pandas_df) * 100:.1f}%")
    
    # Análisis por sexo
    print("\n👥 Distribución por sexo:")
    sex_analysis = pandas_df.groupby(['sex', 'label']).size().unstack(fill_value=0)
    print(sex_analysis)
    
    # Análisis por tipo de trabajo
    print("\n💼 Distribución por tipo de trabajo:")
    work_analysis = pandas_df.groupby(['workclass', 'label']).size().unstack(fill_value=0)
    print(work_analysis)
    
    # Análisis por educación
    print("\n🎓 Distribución por educación:")
    edu_analysis = pandas_df.groupby(['education', 'label']).size().unstack(fill_value=0)
    print(edu_analysis)
    
    # Estadísticas numéricas
    print("\n📈 Estadísticas numéricas:")
    numeric_cols = ['age', 'fnlwgt', 'hours_per_week']
    print(pandas_df[numeric_cols].describe())
    
    return pandas_df

def create_visualizations(pandas_df):
    """
    Crea visualizaciones de los datos
    
    Args:
        pandas_df: DataFrame de Pandas
    """
    print("\n📊 Creando visualizaciones...")
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Análisis Exploratorio de Datos - Clasificación de Ingresos', fontsize=16)
    
    # 1. Distribución de la variable objetivo
    axes[0, 0].pie(pandas_df['label'].value_counts(), 
                   labels=pandas_df['label'].value_counts().index,
                   autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Distribución de Ingresos')
    
    # 2. Distribución por sexo
    sex_label = pandas_df.groupby(['sex', 'label']).size().unstack()
    sex_label.plot(kind='bar', ax=axes[0, 1], stacked=True)
    axes[0, 1].set_title('Distribución por Sexo')
    axes[0, 1].set_xlabel('Sexo')
    axes[0, 1].set_ylabel('Cantidad')
    axes[0, 1].legend(title='Ingresos')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Distribución por tipo de trabajo
    work_label = pandas_df.groupby(['workclass', 'label']).size().unstack()
    work_label.plot(kind='bar', ax=axes[0, 2], stacked=True)
    axes[0, 2].set_title('Distribución por Tipo de Trabajo')
    axes[0, 2].set_xlabel('Tipo de Trabajo')
    axes[0, 2].set_ylabel('Cantidad')
    axes[0, 2].legend(title='Ingresos')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Distribución de edad
    axes[1, 0].hist(pandas_df['age'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Distribución de Edad')
    axes[1, 0].set_xlabel('Edad')
    axes[1, 0].set_ylabel('Frecuencia')
    
    # 5. Distribución de horas por semana
    axes[1, 1].hist(pandas_df['hours_per_week'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Distribución de Horas por Semana')
    axes[1, 1].set_xlabel('Horas por Semana')
    axes[1, 1].set_ylabel('Frecuencia')
    
    # 6. Edad vs Ingresos
    high_income = pandas_df[pandas_df['label'] == '>50K']['age']
    low_income = pandas_df[pandas_df['label'] == '<=50K']['age']
    
    axes[1, 2].hist([low_income, high_income], bins=15, alpha=0.7, 
                   label=['<=50K', '>50K'], color=['red', 'green'])
    axes[1, 2].set_title('Distribución de Edad por Ingresos')
    axes[1, 2].set_xlabel('Edad')
    axes[1, 2].set_ylabel('Frecuencia')
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    # Guardar gráfico
    plt.savefig('results/data_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico guardado en: results/data_analysis.png")
    
    plt.show()

def evaluate_model_performance(predictions_df: DataFrame):
    """
    Evalúa el rendimiento del modelo con métricas detalladas
    
    Args:
        predictions_df: DataFrame con las predicciones
    """
    print("\n📊 EVALUACIÓN DETALLADA DEL MODELO")
    print("=" * 50)
    
    # Convertir a Pandas para análisis más fácil
    pandas_pred = predictions_df.toPandas()
    
    # Calcular matriz de confusión
    from sklearn.metrics import confusion_matrix, classification_report
    
    y_true = pandas_pred['label_indexed']
    y_pred = pandas_pred['prediction']
    
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n🔍 Matriz de Confusión:")
    print("                 Predicción")
    print("                <=50K  >50K")
    print(f"Real <=50K     {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"Real >50K      {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Calcular métricas
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📈 Métricas de Rendimiento:")
    print(f"   • Precisión: {precision:.4f}")
    print(f"   • Sensibilidad (Recall): {recall:.4f}")
    print(f"   • F1-Score: {f1_score:.4f}")
    print(f"   • Exactitud: {accuracy:.4f}")
    
    # Análisis de probabilidades
    print(f"\n🎯 Análisis de Probabilidades:")
    prob_stats = pandas_pred['probability'].apply(lambda x: x[1]).describe()
    print(prob_stats)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': cm
    }

def save_results_to_file(results, filename="results/model_results.txt"):
    """
    Guarda los resultados en un archivo de texto
    
    Args:
        results: Diccionario con los resultados
        filename: Nombre del archivo de salida
    """
    import os
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("RESULTADOS DEL MODELO DE CLASIFICACIÓN DE INGRESOS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("MÉTRICAS DE RENDIMIENTO:\n")
        f.write(f"Precisión: {results['precision']:.4f}\n")
        f.write(f"Sensibilidad (Recall): {results['recall']:.4f}\n")
        f.write(f"F1-Score: {results['f1_score']:.4f}\n")
        f.write(f"Exactitud: {results['accuracy']:.4f}\n\n")
        
        f.write("MATRIZ DE CONFUSIÓN:\n")
        cm = results['confusion_matrix']
        f.write("                 Predicción\n")
        f.write("                <=50K  >50K\n")
        f.write(f"Real <=50K     {cm[0,0]:4d}  {cm[0,1]:4d}\n")
        f.write(f"Real >50K      {cm[1,0]:4d}  {cm[1,1]:4d}\n")
    
    print(f"✅ Resultados guardados en: {filename}")

def create_prediction_summary(predictions_df: DataFrame, spark):
    """
    Crea un resumen de las predicciones
    
    Args:
        predictions_df: DataFrame con las predicciones
        spark: Sesión de Spark
    """
    print("\n📋 RESUMEN DE PREDICCIONES")
    print("=" * 50)
    
    # Contar predicciones por clase
    pred_summary = predictions_df.groupBy("prediction").count().collect()
    
    total_predictions = predictions_df.count()
    
    print(f"Total de predicciones: {total_predictions}")
    
    for row in pred_summary:
        pred_class = ">50K" if row["prediction"] == 1.0 else "<=50K"
        count = row["count"]
        percentage = (count / total_predictions) * 100
        print(f"Predicciones {pred_class}: {count} ({percentage:.1f}%)")
    
    # Análisis de confianza
    print(f"\n🎯 Análisis de Confianza:")
    
    # Convertir a Pandas para análisis de probabilidades
    pandas_pred = predictions_df.toPandas()
    probabilities = pandas_pred['probability'].apply(lambda x: max(x))
    
    print(f"Probabilidad promedio: {probabilities.mean():.3f}")
    print(f"Probabilidad mínima: {probabilities.min():.3f}")
    print(f"Probabilidad máxima: {probabilities.max():.3f}")
    
    # Contar predicciones por nivel de confianza
    high_confidence = (probabilities >= 0.8).sum()
    medium_confidence = ((probabilities >= 0.6) & (probabilities < 0.8)).sum()
    low_confidence = (probabilities < 0.6).sum()
    
    print(f"\nPredicciones por nivel de confianza:")
    print(f"  Alta confianza (≥0.8): {high_confidence} ({high_confidence/total_predictions*100:.1f}%)")
    print(f"  Media confianza (0.6-0.8): {medium_confidence} ({medium_confidence/total_predictions*100:.1f}%)")
    print(f"  Baja confianza (<0.6): {low_confidence} ({low_confidence/total_predictions*100:.1f}%)")
