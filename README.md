# Funcionamiento de la Red Neuronal de 2 Nodos

## 📐 Arquitectura General
```
Entrada (10 características) → NODO 1 → NODO 2 → Salida (Probabilidad)
```

---

## 🔵 NODO 1: Capa Oculta

### Función
Transforma las 10 características de entrada en una representación comprimida de 1 dimensión.

### Operación Matemática
```
z1 = W1 · X + b1
h1 = tanh(z1)
```

Donde:
- **W1**: Matriz de pesos (10 × 1)
- **X**: Vector de entrada (10 características)
- **b1**: Bias escalar
- **tanh**: Función de activación que comprime valores a rango [-1, 1]

### ¿Cuándo se activa?
- Se activa **en cada predicción** (forward pass)
- Se activa **en cada iteración de entrenamiento** (epoch)
- Es el **primer paso** obligatorio antes del Nodo 2

### Salida
- Un valor escalar **h1** en el rango [-1, 1] que representa la "esencia" de los datos de entrada

---

## 🔴 NODO 2: Capa de Salida con Operaciones Matemáticas

### Función
Toma la salida del Nodo 1 y aplica múltiples operaciones matemáticas para producir la probabilidad final.

### Operaciones Matemáticas (en orden)

#### 1. **Combinación Lineal Base**
```
z2_base = W2 · h1 + b2
```

#### 2. **Potencia Ajustable**
```
h1_powered = |h1|^β
```
- **β**: Parámetro aprendible (rango: 0.3 a 3.0)
- Eleva h1 a una potencia variable

#### 3. **Componente Cuadrático**
```
h1_squared = h1²
```

#### 4. **Producto de Hadamard**
```
hadamard = h1 · h1_powered
```
- Multiplicación elemento a elemento

#### 5. **Combinación No Lineal**
```
z2_enhanced = z2_base + α·h1_powered + γ·h1_squared + 0.1·hadamard
```
- **α**: Factor de escalamiento (parámetro aprendible)
- **γ**: Peso del término cuadrático (parámetro aprendible)

#### 6. **Activación Sigmoid**
```
y_pred = sigmoid(z2_enhanced) = 1 / (1 + e^(-z2_enhanced))
```
- Convierte el valor a probabilidad en rango [0, 1]

### ¿Cuándo se activa?
- Se activa **inmediatamente después** del Nodo 1
- Se activa **en cada predicción** y **en cada epoch**
- Es el **segundo y último paso** del forward pass

### Parámetros Aprendibles del Nodo 2
| Parámetro | Función | Valor Inicial |
|-----------|---------|---------------|
| **W2** | Peso lineal | Aleatorio (~0) |
| **b2** | Bias | 0 |
| **α** (alpha) | Escalamiento de potencia | 1.0 |
| **β** (beta) | Exponente de potencia | 1.0 |
| **γ** (gamma) | Peso del término cuadrático | 0.5 |

---

## 🔄 Flujo Completo: Nodo 1 → Nodo 2

### Paso a Paso

1. **Entrada**: Vector X con 10 características
```
   X = [edad, sexo, presión, colesterol, ...]
```

2. **NODO 1 se activa**:
```
   h1 = tanh(W1 · X + b1)
   Resultado: h1 = 0.45 (ejemplo)
```

3. **NODO 2 recibe h1 y aplica operaciones**:
```
   z2_base = W2 · 0.45 + b2 = 0.23
   h1_powered = |0.45|^1.2 = 0.38
   h1_squared = 0.45² = 0.20
   hadamard = 0.45 · 0.38 = 0.17
   
   z2_enhanced = 0.23 + 1.0·0.38 + 0.5·0.20 + 0.1·0.17
               = 0.23 + 0.38 + 0.10 + 0.017
               = 0.727
```

4. **Activación final**:
```
   y_pred = sigmoid(0.727) = 0.674
```

5. **Interpretación**:
```
   Probabilidad de ser fumador = 67.4%
   Predicción final: FUMADOR (> 0.5)
```

---

## ⏱️ Cronología de Activación

### Durante el Entrenamiento (500 épocas)
```
Por cada época (epoch):
  Por cada batch de datos:
    1. NODO 1 se activa → calcula h1
    2. NODO 2 se activa → calcula y_pred
    3. Se calcula el error (loss)
    4. Se calculan gradientes
    5. Se actualizan W1, b1, W2, b2, α, β, γ
```

**Total de activaciones**: ~500 épocas × batches = miles de veces

### Durante la Predicción
```
Por cada muestra a predecir:
  1. NODO 1 se activa → calcula h1
  2. NODO 2 se activa → calcula y_pred
  3. Se retorna la probabilidad
```

---

## 🎯 ¿Por Qué Esta Arquitectura?

### Ventajas del Nodo 1
- **Compresión de información**: Reduce 10 dimensiones a 1
- **Extracción de patrones**: Aprende combinaciones útiles de características
- **No linealidad**: tanh permite capturar relaciones complejas

### Ventajas del Nodo 2 con Operaciones Matemáticas
- **Flexibilidad**: α, β, γ se adaptan durante el entrenamiento
- **No linealidad múltiple**: Combina potencias, cuadrados y productos
- **Mejor separación**: Las operaciones ayudan a distinguir fumadores de no fumadores
- **Focal Loss + Class Weights**: Compensa el desbalance de clases (78% no fumadores vs 22% fumadores)

---

## 🔬 Ejemplo Numérico Completo

### Entrada
```
Persona:
- Edad normalizada: 0.6
- Sexo masculino: 1
- Presión alta: 0
- Colesterol alto: 1
- Triglicéridos altos: 1
- Glucosa alta: 0
- Bebe alcohol: 1
- Obesidad: 0
- Sobrepeso: 1
- Cintura alta: 1
```

### NODO 1
```
z1 = (0.6·0.12 + 1·0.45 + 0·(-0.23) + ... + 1·0.31) + 0.05
   = 0.87
h1 = tanh(0.87) = 0.70
```

### NODO 2
```
z2_base = 0.70 · 0.8 + 0.1 = 0.66
h1_powered = |0.70|^1.15 = 0.65
h1_squared = 0.70² = 0.49
hadamard = 0.70 · 0.65 = 0.46

z2_enhanced = 0.66 + 1.2·0.65 + 0.6·0.49 + 0.1·0.46
            = 0.66 + 0.78 + 0.29 + 0.05
            = 1.78

y_pred = sigmoid(1.78) = 0.856
```

### Resultado
```
Probabilidad de ser fumador: 85.6%
Predicción: FUMADOR ✓
```

---

## 📊 Parámetros Finales Típicos

Después del entrenamiento, los parámetros suelen converger a valores como:

| Parámetro | Valor Final Típico | Interpretación |
|-----------|-------------------|----------------|
| α (alpha) | 1.2 - 1.5 | Amplifica la contribución de h1_powered |
| β (beta) | 0.9 - 1.3 | Controla la no linealidad de la potencia |
| γ (gamma) | 0.4 - 0.7 | Peso del término cuadrático |

---

## 🧮 Resumen Matemático

### Forward Pass Completo
```python
# NODO 1
z1 = W1 · X + b1
h1 = tanh(z1)

# NODO 2
z2_base = W2 · h1 + b2
h1_powered = |h1|^β
h1_squared = h1²
hadamard = h1 · h1_powered
z2 = z2_base + α·h1_powered + γ·h1_squared + 0.1·hadamard
y = sigmoid(z2)
```

### Backward Pass (Entrenamiento)
```python
# Calcular error
loss = FocalLoss(y, y_real) + L2_regularization

# Calcular gradientes
∇W1, ∇b1, ∇W2, ∇b2, ∇α, ∇β, ∇γ = gradient(loss)

# Actualizar parámetros (Adam optimizer)
W1 = W1 - learning_rate · ∇W1
b1 = b1 - learning_rate · ∇b1
W2 = W2 - learning_rate · ∇W2
b2 = b2 - learning_rate · ∇b2
α = α - learning_rate · ∇α
β = β - learning_rate · ∇β
γ = γ - learning_rate · ∇γ
```

---

## ✅ Conclusión

La red funciona como un **pipeline secuencial**:

1. **Nodo 1** comprime y extrae patrones de las 10 características
2. **Nodo 2** toma esa información comprimida y aplica transformaciones no lineales complejas
3. Ambos nodos se **activan en cada predicción** y **aprenden juntos** durante el entrenamiento
4. Los parámetros α, β, γ se **ajustan automáticamente** para maximizar la precisión

El resultado es una red simple (solo 2 nodos) pero poderosa gracias a las **operaciones matemáticas avanzadas** del Nodo 2.
