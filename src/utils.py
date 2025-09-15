"""
Utilities for the income classification project
===============================================

Auxiliary functions for data analysis and visualization
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, when, isnan

def analyze_data_distribution(df: DataFrame, spark):
    """
    Analyze data distribution
    
    Args:
        df: Spark DataFrame
        spark: Spark session
    """
    print("\n📊 ANÁLISIS DE DISTRIBUCIÓN DE DATOS")
    print("=" * 50)
    
    try:
        # Convert to Pandas for easier analysis
        pandas_df = df.toPandas()
        
        # Target variable analysis
        print("\n🎯 Target variable distribution:")
        label_counts = pandas_df['label'].value_counts()
        print(label_counts)
        print(f"Proportion >50K: {label_counts.get('>50K', 0) / len(pandas_df) * 100:.1f}%")
        print(f"Proportion <=50K: {label_counts.get('<=50K', 0) / len(pandas_df) * 100:.1f}%")
        
        # Analysis by sex
        print("\n👥 Distribution by sex:")
        sex_analysis = pandas_df.groupby(['sex', 'label']).size().unstack(fill_value=0)
        print(sex_analysis)
        
        # Analysis by work type
        print("\n💼 Distribution by work type:")
        work_analysis = pandas_df.groupby(['workclass', 'label']).size().unstack(fill_value=0)
        print(work_analysis)
        
        # Analysis by education
        print("\n🎓 Distribution by education:")
        edu_analysis = pandas_df.groupby(['education', 'label']).size().unstack(fill_value=0)
        print(edu_analysis)
        
        # Numeric statistics
        print("\n📈 Numeric statistics:")
        numeric_cols = ['age', 'fnlwgt', 'hours_per_week']
        print(pandas_df[numeric_cols].describe())
        
        return pandas_df
        
    except Exception as e:
        print(f"⚠️  Error in distribution analysis: {e}")
        print("   - Continuing without detailed analysis...")
        return None

def evaluate_model_performance(predictions_df: DataFrame):
    """
    Evaluate model performance with detailed metrics
    
    Args:
        predictions_df: DataFrame with predictions
    """
    print("\n📊 EVALUACIÓN DETALLADA DEL MODELO")
    print("=" * 50)
    
    try:
        # Convert to Pandas for easier analysis
        pandas_pred = predictions_df.toPandas()
        
        # Verify that necessary columns exist
        required_cols = ['label_indexed', 'prediction']
        missing_cols = [col for col in required_cols if col not in pandas_pred.columns]
        
        if missing_cols:
            print(f"⚠️  Missing columns for evaluation: {missing_cols}")
            return None
        
        # Calculate confusion matrix using sklearn
        try:
            from sklearn.metrics import confusion_matrix, classification_report
            
            y_true = pandas_pred['label_indexed']
            y_pred = pandas_pred['prediction']
            
            cm = confusion_matrix(y_true, y_pred)
            
            print("\n🔍 Matriz de Confusión:")
            print("                 Predicción")
            print("                <=50K  >50K")
            print(f"Real <=50K     {cm[0,0]:4d}  {cm[0,1]:4d}")
            print(f"Real >50K      {cm[1,0]:4d}  {cm[1,1]:4d}")
            
            # Calculate metrics
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
            
            if 'probability' in pandas_pred.columns:
                try:
                    print(f"\n🎯 Probability Analysis:")
                    prob_stats = pandas_pred['probability'].apply(lambda x: x[1] if len(x) > 1 else x[0]).describe()
                    print(prob_stats)
                except:
                    print("   - Probabilities not available for detailed analysis")
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'confusion_matrix': cm
            }
            
        except ImportError:
            print("⚠️  sklearn not available, calculating metrics manually...")
            
            y_true = pandas_pred['label_indexed']
            y_pred = pandas_pred['prediction']
            
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            tn = ((y_true == 0) & (y_pred == 0)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            
            accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n📈 Métricas de Rendimiento:")
            print(f"   • Precisión: {precision:.4f}")
            print(f"   • Sensibilidad (Recall): {recall:.4f}")
            print(f"   • F1-Score: {f1_score:.4f}")
            print(f"   • Exactitud: {accuracy:.4f}")
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'confusion_matrix': [[tn, fp], [fn, tp]]
            }
        
    except Exception as e:
        print(f"⚠️  Error in model evaluation: {e}")
        print("   - Basic metrics were shown in the main classifier")
        return None

def save_results_to_file(results, filename="results/model_results.txt"):
    """
    Save results to a text file
    
    Args:
        results: Dictionary with results
        filename: Output file name
    """
    if not results:
        print("⚠️  No results to save")
        return
        
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("RESULTADOS DEL MODELO DE CLASIFICACIÓN DE INGRESOS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("MÉTRICAS DE RENDIMIENTO:\n")
            f.write(f"Precisión: {results['precision']:.4f}\n")
            f.write(f"Sensibilidad (Recall): {results['recall']:.4f}\n")
            f.write(f"F1-Score: {results['f1_score']:.4f}\n")
            f.write(f"Exactitud: {results['accuracy']:.4f}\n\n")
            
            if 'confusion_matrix' in results:
                f.write("MATRIZ DE CONFUSIÓN:\n")
                cm = results['confusion_matrix']
                if isinstance(cm, list):
                    # List format
                    f.write("                 Predicción\n")
                    f.write("                <=50K  >50K\n")
                    f.write(f"Real <=50K     {cm[0][0]:4d}  {cm[0][1]:4d}\n")
                    f.write(f"Real >50K      {cm[1][0]:4d}  {cm[1][1]:4d}\n")
                else:
                    # Numpy array format
                    f.write("                 Predicción\n")
                    f.write("                <=50K  >50K\n")
                    f.write(f"Real <=50K     {cm[0,0]:4d}  {cm[0,1]:4d}\n")
                    f.write(f"Real >50K      {cm[1,0]:4d}  {cm[1,1]:4d}\n")
        
        print(f"✅ Resultados guardados en: {filename}")
        
    except Exception as e:
        print(f"⚠️  Error saving results: {e}")

def create_prediction_summary(predictions_df: DataFrame, spark):
    """
    Create prediction summary
    
    Args:
        predictions_df: DataFrame with predictions
        spark: Spark session
    """
    print("\n📋 RESUMEN DE PREDICCIONES")
    print("=" * 50)
    
    try:
        # Count predictions by class
        pred_summary = predictions_df.groupBy("prediction").count().collect()
        
        total_predictions = predictions_df.count()
        
        print(f"Total de predicciones: {total_predictions}")
        
        for row in pred_summary:
            pred_class = ">50K" if row["prediction"] == 1.0 else "<=50K"
            count = row["count"]
            percentage = (count / total_predictions) * 100
            print(f"Predicciones {pred_class}: {count} ({percentage:.1f}%)")
        
        # Confidence analysis if probabilities are available
        try:
            print(f"\n🎯 Confidence Analysis:")
            
            # Convert to Pandas for probability analysis
            pandas_pred = predictions_df.toPandas()
            if 'probability' in pandas_pred.columns:
                probabilities = pandas_pred['probability'].apply(lambda x: max(x) if isinstance(x, list) else max(x.toArray()))
                
                print(f"Average probability: {probabilities.mean():.3f}")
                print(f"Minimum probability: {probabilities.min():.3f}")
                print(f"Maximum probability: {probabilities.max():.3f}")
                
                # Count predictions by confidence level
                high_confidence = (probabilities >= 0.8).sum()
                medium_confidence = ((probabilities >= 0.6) & (probabilities < 0.8)).sum()
                low_confidence = (probabilities < 0.6).sum()
                
                print(f"\nPredictions by confidence level:")
                print(f"  High confidence (≥0.8): {high_confidence} ({high_confidence/total_predictions*100:.1f}%)")
                print(f"  Medium confidence (0.6-0.8): {medium_confidence} ({medium_confidence/total_predictions*100:.1f}%)")
                print(f"  Low confidence (<0.6): {low_confidence} ({low_confidence/total_predictions*100:.1f}%)")
            else:
                print("   - Probability information not available")
                
        except Exception as e:
            print(f"   - Error in confidence analysis: {e}")
        
    except Exception as e:
        print(f"⚠️  Error in prediction summary: {e}")
        print("   - Continuing without detailed summary...")

def create_simple_visualization(pandas_df):
    """
    Create simple visualizations if matplotlib is available
    
    Args:
        pandas_df: Pandas DataFrame with data
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Target variable distribution
        pandas_df['label'].value_counts().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Income Distribution')
        axes[0,0].set_ylabel('Frequency')
        
        # Age distribution
        pandas_df['age'].hist(bins=20, ax=axes[0,1])
        axes[0,1].set_title('Age Distribution')
        axes[0,1].set_xlabel('Age')
        axes[0,1].set_ylabel('Frequency')
        
        # Hours per week
        pandas_df['hours_per_week'].hist(bins=20, ax=axes[1,0])
        axes[1,0].set_title('Hours per Week')
        axes[1,0].set_xlabel('Hours')
        axes[1,0].set_ylabel('Frequency')
        
        # Income by sex
        crosstab = pd.crosstab(pandas_df['sex'], pandas_df['label'])
        crosstab.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Income by Sex')
        axes[1,1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save if possible
        try:
            os.makedirs('results', exist_ok=True)
            plt.savefig('results/data_analysis.png', dpi=300, bbox_inches='tight')
            print("📊 Charts saved to: results/data_analysis.png")
        except:
            print("📊 Charts created (could not be saved)")
            
        plt.show()
        
    except ImportError:
        print("⚠️  matplotlib/seaborn not available - skipping visualizations")
    except Exception as e:
        print(f"⚠️  Error in visualization: {e}")
