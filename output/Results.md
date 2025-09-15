# 📊 Resultados del Modelo de Clasificación de Ingresos

## 📈 Métricas de Rendimiento

| Métrica | Valor |
|---------|-------|
| **Precisión** | 0.5243 |
| **Sensibilidad (Recall)** | 0.3655 |
| **F1-Score** | 0.4307 |
| **Exactitud** | 0.5295 |

## 🔍 Matriz de Confusión

```
                 Predicción
                <=50K  >50K
Real <=50K      703   323
Real >50K       618   356
```

### 📋 Interpretación

- **Verdaderos Negativos (TN)**: 703 casos <=50K clasificados correctamente
- **Falsos Positivos (FP)**: 323 casos <=50K clasificados incorrectamente como >50K
- **Falsos Negativos (FN)**: 618 casos >50K clasificados incorrectamente como <=50K
- **Verdaderos Positivos (TP)**: 356 casos >50K clasificados correctamente

