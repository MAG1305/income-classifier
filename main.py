"""
Script Principal - Clasificador de Ingresos con Spark ML
=======================================================

Este es el punto de entrada principal para ejecutar todo el análisis
de clasificación de ingresos.

Uso:
    python main.py
"""

import sys
import os
from pyspark.sql import SparkSession

# Agregar directorios al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

from income_classifier import IncomeClassifier
from utils import analyze_data_distribution, create_visualizations, evaluate_model_performance, save_results_to_file, create_prediction_summary
from spark_config import create_spark_session, stop_spark_session

def main():
    """
    Función principal que ejecuta todo el análisis
    """
    print("=" * 80)
    print("🏦 CLASIFICADOR DE INGRESOS CON SPARK ML - ANÁLISIS COMPLETO")
    print("=" * 80)
    
    data_path = "data/adult_income_sample.csv"
    spark = None
    
    try:
        print("\n🚀 Inicializando Spark...")
        spark = create_spark_session("IncomeClassifierComplete")
        print("✅ Spark inicializado correctamente")
        
        classifier = IncomeClassifier(data_path)
        classifier.spark = spark 
        
        print("\n📊 Cargando y analizando datos...")
        classifier.load_data()
        
        print("\n🔍 Realizando análisis exploratorio...")
        pandas_df = analyze_data_distribution(classifier.df, spark)
        
        print("\n📊 Creando visualizaciones...")
        try:
            create_visualizations(pandas_df)
        except Exception as e:
            print(f"⚠️  No se pudieron crear las visualizaciones: {e}")
            print("   (Esto es normal si no tienes matplotlib instalado)")
        
        print("\n🔧 Preprocesando datos...")
        preprocessing_stages = classifier.preprocess_data()
        
        print("\n🤖 Creando y entrenando modelo...")
        classifier.create_model(preprocessing_stages)
        predictions = classifier.train_model()
        
        print("\n📈 Evaluando rendimiento del modelo...")
        results = evaluate_model_performance(predictions)
        
        print("\n💾 Guardando resultados...")
        save_results_to_file(results)
        
        create_prediction_summary(predictions, spark)
        
        print("\n🆕 Creando datos nuevos para predicción...")
        new_df = classifier.create_new_data()
        new_predictions = classifier.predict_new_data(new_df)
        
        print("\n🎉 ANÁLISIS COMPLETO FINALIZADO EXITOSAMENTE!")
        print("\n📁 Archivos generados:")
        print("   • results/model_results.txt - Métricas del modelo")
        print("   • results/data_analysis.png - Gráficos de análisis (si está disponible)")
        
        print("\n📊 Resumen del análisis:")
        print(f"   • Dataset: {classifier.df.count()} registros")
        print(f"   • Precisión del modelo: {results['precision']:.3f}")
        print(f"   • Sensibilidad: {results['recall']:.3f}")
        print(f"   • F1-Score: {results['f1_score']:.3f}")
        print(f"   • Exactitud: {results['accuracy']:.3f}")
        
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Detener Spark
        if spark:
            stop_spark_session(spark)
            print("\n🛑 Sesión de Spark detenida")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
