"""
Configuración de Spark para el proyecto de clasificación de ingresos
"""

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

def create_spark_session(app_name="IncomeClassifier"):
    """
    Crea y configura una sesión de Spark
    
    Args:
        app_name (str): Nombre de la aplicación Spark
        
    Returns:
        SparkSession: Sesión configurada de Spark
    """
    conf = SparkConf()
    conf.set("spark.sql.adaptive.enabled", "true")
    conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    
    spark = SparkSession.builder \
        .appName(app_name) \
        .config(conf=conf) \
        .getOrCreate()
    
    # Configurar nivel de logging para reducir verbosidad
    spark.sparkContext.setLogLevel("WARN")
    
    return spark

def stop_spark_session(spark):
    """
    Detiene la sesión de Spark
    
    Args:
        spark (SparkSession): Sesión de Spark a detener
    """
    spark.stop()
