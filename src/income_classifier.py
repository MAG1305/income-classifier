"""
Clasificador de Ingresos con Spark ML
=====================================

Este script implementa un clasificador binario para predecir si una persona
gana más de 50K al año usando regresión logística con Spark ML.

Autor: DataPros
Fecha: 2024
"""

import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, count
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from spark_config import create_spark_session, stop_spark_session

class IncomeClassifier:
    """
    Clasificador de ingresos usando Spark ML
    """
    
    def __init__(self, data_path):
        """
        Inicializa el clasificador
        
        Args:
            data_path (str): Ruta al archivo CSV con los datos
        """
        self.data_path = data_path
        self.spark = None
        self.df = None
        self.pipeline = None
        self.model = None
        
    def initialize_spark(self):
        """Inicializa la sesión de Spark"""
        print("🚀 Inicializando Spark...")
        self.spark = create_spark_session("IncomeClassifier")
        print("✅ Spark inicializado correctamente")
        
    def load_data(self):
        """
        Carga los datos desde el archivo CSV
        
        Tarea 1: Carga de datos
        """
        print("\n📊 Cargando datos...")
        
        schema = StructType([
            StructField("age", IntegerType(), True),
            StructField("sex", StringType(), True),
            StructField("workclass", StringType(), True),
            StructField("fnlwgt", IntegerType(), True),
            StructField("education", StringType(), True),
            StructField("hours_per_week", IntegerType(), True),
            StructField("label", StringType(), True)
        ])
        
        self.df = self.spark.read \
            .option("header", "true") \
            .option("inferSchema", "false") \
            .schema(schema) \
            .csv(self.data_path)
        
        print(f"✅ Datos cargados: {self.df.count()} registros")
        
        print("\n📋 Esquema de los datos:")
        self.df.printSchema()
        
        print("\n📋 Primeros 10 registros:")
        self.df.show(10, truncate=False)
        
        print("\n📈 Estadísticas descriptivas:")
        self.df.describe().show()
        
        print("\n🔍 Verificación de valores nulos:")
        # Verificar valores nulos (solo para columnas numéricas)
        numeric_columns = ["age", "fnlwgt", "hours_per_week"]
        string_columns = ["sex", "workclass", "education", "label"]
        
        # Para columnas numéricas: verificar nulos y NaN
        numeric_nulls = self.df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in numeric_columns])
        print("Columnas numéricas (nulos y NaN):")
        numeric_nulls.show()
        
        # Para columnas de string: solo verificar nulos
        string_nulls = self.df.select([count(when(col(c).isNull(), c)).alias(c) for c in string_columns])
        print("Columnas de texto (nulos):")
        string_nulls.show()
        
    def preprocess_data(self):
        """
        Preprocesa las variables categóricas
        
        Tarea 2: Preprocesamiento de variables categóricas
        """
        print("\n🔧 Preprocesando variables categóricas...")
        
        # Variables categóricas a procesar
        categorical_columns = ["sex", "workclass", "education", "label"]
        
        # Crear StringIndexers para cada variable categórica
        string_indexers = []
        for col_name in categorical_columns:
            indexer = StringIndexer(
                inputCol=col_name,
                outputCol=f"{col_name}_indexed",
                handleInvalid="keep"
            )
            string_indexers.append(indexer)
        
        # Crear OneHotEncoders para las variables categóricas (excepto label)
        one_hot_encoders = []
        for col_name in categorical_columns[:-1]:  # Excluir 'label'
            encoder = OneHotEncoder(
                inputCol=f"{col_name}_indexed",
                outputCol=f"{col_name}_encoded"
            )
            one_hot_encoders.append(encoder)
        
        # Crear VectorAssembler para combinar todas las características
        # Tarea 3: Ensamblaje de características
        feature_columns = ["age", "fnlwgt", "hours_per_week"]
        encoded_columns = [f"{col}_encoded" for col in categorical_columns[:-1]]
        all_features = feature_columns + encoded_columns
        
        vector_assembler = VectorAssembler(
            inputCols=all_features,
            outputCol="features"
        )
        
        # Crear el pipeline de preprocesamiento
        preprocessing_stages = string_indexers + one_hot_encoders + [vector_assembler]
        
        print("✅ Preprocesamiento configurado")
        print(f"   - Variables categóricas indexadas: {categorical_columns}")
        print(f"   - Variables numéricas: {feature_columns}")
        print(f"   - Características finales: {all_features}")
        
        return preprocessing_stages
    
    def create_model(self, preprocessing_stages):
        """
        Crea y configura el modelo de regresión logística
        
        Tarea 4: Definición y entrenamiento del modelo
        """
        print("\n🤖 Configurando modelo de Regresión Logística...")
        
        # Configurar regresión logística
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label_indexed",
            maxIter=100,
            regParam=0.01,
            elasticNetParam=0.8
        )
        
        # Crear pipeline completo
        self.pipeline = Pipeline(stages=preprocessing_stages + [lr])
        
        print("✅ Modelo configurado")
        print("   - Algoritmo: Regresión Logística")
        print("   - Máximo de iteraciones: 100")
        print("   - Parámetro de regularización: 0.01")
        print("   - Elastic Net: 0.8")
        
    def train_model(self):
        """
        Entrena el modelo con todos los datos
        
        Tarea 5: Evaluación del modelo
        """
        print("\n🎯 Entrenando modelo...")
        
        # Entrenar el modelo
        self.model = self.pipeline.fit(self.df)
        
        print("✅ Modelo entrenado exitosamente")
        
        # Hacer predicciones
        predictions = self.model.transform(self.df)
        
        # Mostrar predicciones con probabilidades
        print("\n📊 Predicciones del modelo:")
        predictions.select(
            "age", "sex", "workclass", "education", "hours_per_week",
            "label", "label_indexed", "prediction", "probability"
        ).show(20, truncate=False)
        
        # Calcular métricas de evaluación
        evaluator = BinaryClassificationEvaluator(
            labelCol="label_indexed",
            rawPredictionCol="rawPrediction"
        )
        
        auc = evaluator.evaluate(predictions)
        print(f"\n📈 Métricas de evaluación:")
        print(f"   - AUC: {auc:.4f}")
        
        # Análisis de resultados
        print("\n🔍 Análisis de resultados:")
        print("   - El modelo ha sido entrenado con 2000 registros")
        print("   - Se utilizaron características demográficas y laborales")
        print("   - Las predicciones muestran la probabilidad de ganar >50K")
        
        return predictions
    
    def create_new_data(self):
        """
        Crea datos nuevos para predicción
        
        Tarea 6: Predicción con nuevos datos
        """
        print("\n🆕 Creando datos nuevos para predicción...")
        
        # Crear DataFrame con datos nuevos
        new_data = [
            (25, "Male", "Private", 150000, "Bachelors", 40),  # Joven con educación universitaria
            (45, "Female", "Gov", 200000, "Masters", 35),      # Mujer con maestría en gobierno
            (30, "Male", "Self-emp", 180000, "HS-grad", 50),   # Hombre autoempleado
            (55, "Female", "Private", 250000, "Bachelors", 30), # Mujer mayor con experiencia
            (22, "Male", "Private", 120000, "Some-college", 45), # Joven con educación parcial
            (40, "Female", "Gov", 220000, "Masters", 25),      # Mujer con maestría, pocas horas
            (35, "Male", "Self-emp", 300000, "Bachelors", 60), # Hombre autoempleado, muchas horas
            (50, "Female", "Private", 280000, "HS-grad", 40),  # Mujer con experiencia
            (28, "Male", "Gov", 160000, "Bachelors", 35)       # Hombre joven en gobierno
        ]
        
        schema = StructType([
            StructField("age", IntegerType(), True),
            StructField("sex", StringType(), True),
            StructField("workclass", StringType(), True),
            StructField("fnlwgt", IntegerType(), True),
            StructField("education", StringType(), True),
            StructField("hours_per_week", IntegerType(), True)
        ])
        
        new_df = self.spark.createDataFrame(new_data, schema)
        
        print("✅ Datos nuevos creados (9 registros)")
        print("\n👀 Datos nuevos:")
        new_df.show(truncate=False)
        
        return new_df
    
    def predict_new_data(self, new_df):
        """
        Realiza predicciones con los datos nuevos
        """
        print("\n🔮 Realizando predicciones con datos nuevos...")
        
        # Hacer predicciones
        predictions = self.model.transform(new_df)
        
        # Mostrar resultados
        print("\n📊 Predicciones para datos nuevos:")
        results = predictions.select(
            "age", "sex", "workclass", "education", "hours_per_week",
            "prediction", "probability"
        ).collect()
        
        for i, row in enumerate(results, 1):
            prob = row["probability"][1]  # Probabilidad de clase >50K
            pred = ">50K" if row["prediction"] == 1.0 else "<=50K"
            print(f"\nPersona {i}:")
            print(f"  Edad: {row['age']}, Sexo: {row['sex']}, Trabajo: {row['workclass']}")
            print(f"  Educación: {row['education']}, Horas/semana: {row['hours_per_week']}")
            print(f"  Predicción: {pred} (Probabilidad: {prob:.3f})")
        
        return predictions
    
    def run_complete_analysis(self):
        """
        Ejecuta el análisis completo
        """
        try:
            # Inicializar Spark
            self.initialize_spark()
            
            # Cargar datos
            self.load_data()
            
            # Preprocesar datos
            preprocessing_stages = self.preprocess_data()
            
            # Crear modelo
            self.create_model(preprocessing_stages)
            
            # Entrenar modelo
            predictions = self.train_model()
            
            # Crear datos nuevos
            new_df = self.create_new_data()
            
            # Predecir datos nuevos
            new_predictions = self.predict_new_data(new_df)
            
            print("\n🎉 Análisis completado exitosamente!")
            
        except Exception as e:
            print(f"\n❌ Error durante el análisis: {str(e)}")
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
    print("🏦 CLASIFICADOR DE INGRESOS CON SPARK ML")
    print("=" * 60)
    
    # Ruta al archivo de datos
    data_path = "data/adult_income_sample.csv"
    
    # Crear y ejecutar clasificador
    classifier = IncomeClassifier(data_path)
    classifier.run_complete_analysis()

if __name__ == "__main__":
    main()
