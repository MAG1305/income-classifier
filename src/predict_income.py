"""
Script de Predicción de Ingresos
================================

Script independiente para realizar predicciones con datos nuevos
usando un modelo previamente entrenado.

Uso:
    python predict_income.py
"""

import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml import PipelineModel

# Agregar el directorio config al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from spark_config import create_spark_session, stop_spark_session

class IncomePredictor:
    """
    Clasificador para realizar predicciones con datos nuevos
    """
    
    def __init__(self, model_path=None):
        """
        Inicializa el predictor
        
        Args:
            model_path (str): Ruta al modelo guardado (opcional)
        """
        self.model_path = model_path
        self.spark = None
        self.model = None
        
    def initialize_spark(self):
        """Inicializa la sesión de Spark"""
        print("🚀 Inicializando Spark...")
        self.spark = create_spark_session("IncomePredictor")
        print("✅ Spark inicializado")
        
    def load_model(self, model_path):
        """
        Carga un modelo previamente entrenado
        
        Args:
            model_path (str): Ruta al modelo guardado
        """
        print(f"📦 Cargando modelo desde: {model_path}")
        self.model = PipelineModel.load(model_path)
        print("✅ Modelo cargado exitosamente")
        
    def create_sample_data(self):
        """
        Crea datos de muestra para predicción
        """
        print("\n🆕 Creando datos de muestra...")
        
        # Datos de ejemplo más diversos
        sample_data = [
            # Casos que probablemente ganen >50K
            (35, "Male", "Private", 200000, "Masters", 45),    # Hombre con maestría
            (42, "Female", "Gov", 180000, "Bachelors", 40),    # Mujer con bachillerato en gobierno
            (28, "Male", "Self-emp", 250000, "Bachelors", 55), # Hombre autoempleado, muchas horas
            
            # Casos que probablemente ganen <=50K
            (22, "Female", "Private", 120000, "HS-grad", 30),  # Mujer joven, pocas horas
            (19, "Male", "Private", 100000, "11th", 25),       # Hombre muy joven, sin educación completa
            (60, "Female", "Gov", 150000, "Some-college", 20), # Mujer mayor, pocas horas
            
            # Casos intermedios
            (30, "Male", "Private", 160000, "Assoc", 35),      # Hombre con educación técnica
            (45, "Female", "Self-emp", 220000, "HS-grad", 50), # Mujer autoempleada, muchas horas
            (25, "Female", "Gov", 140000, "Bachelors", 30),    # Mujer joven con bachillerato
        ]
        
        schema = StructType([
            StructField("age", IntegerType(), True),
            StructField("sex", StringType(), True),
            StructField("workclass", StringType(), True),
            StructField("fnlwgt", IntegerType(), True),
            StructField("education", StringType(), True),
            StructField("hours_per_week", IntegerType(), True)
        ])
        
        df = self.spark.createDataFrame(sample_data, schema)
        
        print("✅ Datos de muestra creados (9 registros)")
        print("\n👀 Datos de muestra:")
        df.show(truncate=False)
        
        return df
    
    def predict(self, df):
        """
        Realiza predicciones con el modelo
        
        Args:
            df: DataFrame con los datos a predecir
        """
        print("\n🔮 Realizando predicciones...")
        
        # Hacer predicciones
        predictions = self.model.transform(df)
        
        # Mostrar resultados detallados
        print("\n📊 Resultados de las predicciones:")
        print("=" * 80)
        
        results = predictions.select(
            "age", "sex", "workclass", "education", "hours_per_week",
            "prediction", "probability"
        ).collect()
        
        for i, row in enumerate(results, 1):
            prob_high = row["probability"][1]  # Probabilidad de >50K
            prob_low = row["probability"][0]   # Probabilidad de <=50K
            pred = ">50K" if row["prediction"] == 1.0 else "<=50K"
            
            print(f"\n👤 Persona {i}:")
            print(f"   📋 Perfil:")
            print(f"      • Edad: {row['age']} años")
            print(f"      • Sexo: {row['sex']}")
            print(f"      • Tipo de trabajo: {row['workclass']}")
            print(f"      • Educación: {row['education']}")
            print(f"      • Horas por semana: {row['hours_per_week']}")
            print(f"   🎯 Predicción: {pred}")
            print(f"   📊 Probabilidades:")
            print(f"      • <=50K: {prob_low:.3f} ({prob_low*100:.1f}%)")
            print(f"      • >50K:  {prob_high:.3f} ({prob_high*100:.1f}%)")
            print("-" * 80)
        
        return predictions
    
    def analyze_predictions(self, predictions):
        """
        Analiza los resultados de las predicciones
        
        Args:
            predictions: DataFrame con las predicciones
        """
        print("\n📈 Análisis de las predicciones:")
        
        # Contar predicciones por clase
        pred_counts = predictions.groupBy("prediction").count().collect()
        
        high_income_count = 0
        low_income_count = 0
        
        for row in pred_counts:
            if row["prediction"] == 1.0:
                high_income_count = row["count"]
            else:
                low_income_count = row["count"]
        
        total = high_income_count + low_income_count
        
        print(f"   • Personas predichas con ingresos >50K: {high_income_count} ({high_income_count/total*100:.1f}%)")
        print(f"   • Personas predichas con ingresos <=50K: {low_income_count} ({low_income_count/total*100:.1f}%)")
        
        # Análisis por características
        print(f"\n🔍 Análisis por características:")
        
        # Por sexo
        print("   Por sexo:")
        sex_analysis = predictions.groupBy("sex", "prediction").count().collect()
        for row in sex_analysis:
            sex = row["sex"]
            pred = ">50K" if row["prediction"] == 1.0 else "<=50K"
            count = row["count"]
            print(f"      • {sex} predicho como {pred}: {count}")
        
        # Por tipo de trabajo
        print("   Por tipo de trabajo:")
        work_analysis = predictions.groupBy("workclass", "prediction").count().collect()
        for row in work_analysis:
            work = row["workclass"]
            pred = ">50K" if row["prediction"] == 1.0 else "<=50K"
            count = row["count"]
            print(f"      • {work} predicho como {pred}: {count}")
    
    def run_prediction_demo(self):
        """
        Ejecuta una demostración de predicción
        """
        try:
            # Inicializar Spark
            self.initialize_spark()
            
            # Crear datos de muestra
            sample_df = self.create_sample_data()
            
            # Nota: En un escenario real, aquí cargarías un modelo previamente entrenado
            print("\n⚠️  NOTA: Este es un script de demostración.")
            print("   Para usar un modelo real, primero ejecuta 'income_classifier.py'")
            print("   para entrenar y guardar el modelo, luego modifica este script")
            print("   para cargar el modelo guardado.")
            
            print("\n🎯 Simulando predicciones...")
            print("   (En un escenario real, aquí se cargaría el modelo entrenado)")
            
        except Exception as e:
            print(f"\n❌ Error durante la predicción: {str(e)}")
            raise
            
        finally:
            # Detener Spark
            if self.spark:
                stop_spark_session(self.spark)
                print("\n🛑 Sesión de Spark detenida")

def main():
    """
    Función principal
    """
    print("=" * 60)
    print("🔮 PREDICTOR DE INGRESOS - DATOS NUEVOS")
    print("=" * 60)
    
    # Crear predictor
    predictor = IncomePredictor()
    
    # Ejecutar demostración
    predictor.run_prediction_demo()

if __name__ == "__main__":
    main()
