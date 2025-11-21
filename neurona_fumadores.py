import tensorflow as tf
import numpy as np
from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# CONEXIÓN Y CARGA DE DATOS
# ============================================

def conectar_mongodb(uri=None, db_name="salud_db"):
    try:
        if uri is None:
            uri = "mongodb+srv://2022371082_db_user:marianahernandezdimas15102004@cluster0.gtmppy1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        
        client = MongoClient(uri)
        client.admin.command('ping')
        
        db = client[db_name]
        print(f"Conexión exitosa a MongoDB Atlas - Base de datos: {db_name}")
        return db
    except Exception as e:
        print(f"Error al conectar: {e}")
        return None

def cargar_y_procesar_datos(db):
    try:
        print("\nCargando datos desde 'encuestas_fumadores'...")
        
        encuestas = db.encuestas_fumadores.find()
        datos = list(encuestas)
        
        if not datos:
            print("No hay datos en 'encuestas_fumadores'")
            return None
        
        df = pd.DataFrame(datos)
        print(f"Datos cargados: {len(df)} registros")
        
        df_procesado = pd.DataFrame()
        
        if 'SMK_stat_type_cd' in df.columns:
            df_procesado['fuma'] = (df['SMK_stat_type_cd'] == 3).astype(int)
            fumadores = df_procesado['fuma'].sum()
            print(f"'fuma': Creada ({fumadores} fumadores)")
        else:
            return None
        
        if 'sex' in df.columns:
            df_procesado['sexo_masculino'] = (df['sex'] == 'Male').astype(int)
        
        if 'age' in df.columns:
            edad_min, edad_max = df['age'].min(), df['age'].max()
            df_procesado['edad_normalizada'] = (df['age'] - edad_min) / (edad_max - edad_min)
        
        if 'SBP' in df.columns:
            df_procesado['presion_alta'] = (df['SBP'] > 140).astype(int)
        
        if 'tot_chole' in df.columns:
            df_procesado['colesterol_alto'] = (df['tot_chole'] > 200).astype(int)
        
        if 'triglyceride' in df.columns:
            df_procesado['trigliceridos_altos'] = (df['triglyceride'] > 150).astype(int)
        
        if 'BLDS' in df.columns:
            df_procesado['glucosa_alta'] = (df['BLDS'] > 100).astype(int)
        
        if 'DRK_YN' in df.columns:
            df_procesado['bebe_alcohol'] = (df['DRK_YN'] == 'Y').astype(int)
        
        if 'height' in df.columns and 'weight' in df.columns:
            altura_m = df['height'] / 100
            imc = df['weight'] / (altura_m ** 2)
            df_procesado['tiene_obesidad'] = (imc > 30).astype(int)
            df_procesado['tiene_sobrepeso'] = (imc > 25).astype(int)
        
        if 'waistline' in df.columns and 'sex' in df.columns:
            cintura_alta = []
            for _, row in df.iterrows():
                if row['sex'] == 'Male':
                    cintura_alta.append(1 if row['waistline'] > 90 else 0)
                else:
                    cintura_alta.append(1 if row['waistline'] > 85 else 0)
            df_procesado['cintura_alta'] = cintura_alta
        
        df_procesado = df_procesado.dropna()
        
        print(f"Procesamiento completado: {len(df_procesado)} registros válidos")
        
        return df_procesado
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def preparar_dataset(df_procesado):
    caracteristicas = [col for col in df_procesado.columns if col != 'fuma']
    
    if not caracteristicas:
        return None, None, None, None
    
    X = df_procesado[caracteristicas].values.astype(np.float32)
    y = df_procesado['fuma'].values.astype(np.float32)
    
    print(f"\nDataset: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"Fumadores: {int(y.sum())} ({(y.sum()/len(y))*100:.1f}%)")
    
    # CAMBIO: Pesos más conservadores para evitar predicción constante
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y)
    class_weights = class_weights * 0.6  # Suavizar más agresivamente
    
    print(f"\n⚖️ PESOS AJUSTADOS:")
    print(f"  No Fumador: {class_weights[0]:.3f}")
    print(f"  Fumador:    {class_weights[1]:.3f}")
    
    return X, y, caracteristicas, class_weights

# ============================================
# RED NEURONAL CON EXACTAMENTE 3 NODOS
# ============================================

class RedTresNodosMejorada:
    """
    Red con EXACTAMENTE 3 NODOS:
    - NODO 1: Capa oculta (2 neuronas) - Extrae patrones
    - NODO 2: Capa intermedia (1 neurona) - Combina patrones
    - NODO 3: Capa salida (1 neurona) - Operaciones matemáticas con pesos ajustables
    """
    
    def __init__(self, num_entradas, class_weights, learning_rate=0.001):
        self.num_entradas = num_entradas
        self.learning_rate = learning_rate
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        
        # Inicialización He
        limit = np.sqrt(2.0 / num_entradas)
        
        # NODO 1: Capa oculta (2 NEURONAS)
        self.W1 = tf.Variable(
            tf.random.uniform([num_entradas, 2], -limit, limit),
            name='W1_nodo1', trainable=True
        )
        self.b1 = tf.Variable(tf.zeros([2]), name='b1_nodo1', trainable=True)
        
        # NODO 2: Capa intermedia (1 NEURONA)
        self.W2 = tf.Variable(
            tf.random.uniform([2, 1], -limit, limit),
            name='W2_nodo2', trainable=True
        )
        self.b2 = tf.Variable(tf.zeros([1]), name='b2_nodo2', trainable=True)
        
        # NODO 3: Capa salida (1 NEURONA) con operaciones matemáticas
        self.W3 = tf.Variable(
            tf.random.uniform([1, 1], -limit, limit),
            name='W3_nodo3', trainable=True
        )
        self.b3 = tf.Variable(tf.zeros([1]), name='b3_nodo3', trainable=True)
        
        # Parámetros de operaciones matemáticas en NODO 3 (PESOS AJUSTABLES)
        self.alpha = tf.Variable(1.0, name='alpha_nodo3', trainable=True)
        self.beta = tf.Variable(1.0, name='beta_nodo3', trainable=True)
        self.gamma = tf.Variable(0.5, name='gamma_nodo3', trainable=True)
        self.delta = tf.Variable(0.3, name='delta_nodo3', trainable=True)
        
        self.historial_perdida = []
        self.historial_precision = []
        self.historial_perdida_val = []
        self.historial_precision_val = []
        self.historial_f1_val = []
        self.historial_operaciones = []
        
        print(f"\n🧠 RED CON EXACTAMENTE 3 NODOS")
        print(f"="*60)
        print(f"NODO 1 (Oculta):     {num_entradas} entradas → 2 neuronas (tanh)")
        print(f"NODO 2 (Intermedia): 2 entradas → 1 neurona (ReLU)")
        print(f"NODO 3 (Salida):     1 entrada → 1 neurona + operaciones matemáticas")
        print(f"  • Pesos ajustables en Nodo 3: α, β, γ, δ")
        print(f"  • Operaciones: potencias, productos, exponenciales")
        print(f"Learning rate: {learning_rate}")
        print(f"="*60)
    
    def forward(self, X):
        """Forward pass con EXACTAMENTE 3 NODOS"""
        
        # NODO 1: Extracción de características (2 neuronas)
        z1 = tf.matmul(X, self.W1) + self.b1
        h1 = tf.nn.tanh(z1)  # Salida: (batch_size, 2)
        
        # NODO 2: Combinación intermedia (1 neurona)
        z2 = tf.matmul(h1, self.W2) + self.b2
        h2 = tf.nn.relu(z2)  # ReLU para mantener valores positivos
        
        # NODO 3: Operaciones matemáticas avanzadas con PESOS AJUSTABLES
        
        # Operación 1: Combinación lineal base
        z3_base = tf.matmul(h2, self.W3) + self.b3
        
        # Operación 2: Potencia ajustable (con peso β)
        beta_clip = tf.clip_by_value(self.beta, 0.3, 3.0)
        h2_abs = tf.abs(h2)
        h2_powered = tf.pow(h2_abs + 1e-7, beta_clip)
        
        # Operación 3: Componente cuadrático (con peso γ)
        h2_squared = tf.square(h2)
        
        # Operación 4: Exponencial suavizada (con peso δ)
        h2_exp = tf.exp(tf.clip_by_value(h2 * 0.5, -2, 2))
        
        # Operación 5: Producto de Hadamard
        hadamard = h2 * h2_powered
        
        # Operación 6: Combinación no lineal con TODOS los pesos ajustables
        z3_enhanced = (
            z3_base + 
            self.alpha * h2_powered +
            self.gamma * h2_squared +
            self.delta * h2_exp +
            0.1 * hadamard
        )
        
        # Operación 7: Sigmoid final
        y_pred = tf.sigmoid(z3_enhanced)
        
        return y_pred, h1, h2, h2_powered, h2_squared, h2_exp, hadamard
    
    def calcular_perdida_focal(self, y_pred, y_real, gamma=2.0):
        """Focal Loss para manejar desbalance"""
        epsilon = 1e-7
        y_pred_clip = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        
        # Focal loss
        p_t = tf.where(tf.equal(y_real, 1.0), y_pred_clip, 1 - y_pred_clip)
        focal_weight = tf.pow(1 - p_t, gamma)
        
        bce = -(
            y_real * tf.math.log(y_pred_clip) + 
            (1 - y_real) * tf.math.log(1 - y_pred_clip)
        )
        
        focal_loss = focal_weight * bce
        
        # Pesos por clase
        weights = tf.where(
            tf.equal(y_real, 1.0),
            self.class_weights[1],
            self.class_weights[0]
        )
        
        weighted_loss = focal_loss * weights
        
        # Regularización L2
        l2_loss = 0.0001 * (
            tf.reduce_sum(tf.square(self.W1)) + 
            tf.reduce_sum(tf.square(self.W2)) +
            tf.reduce_sum(tf.square(self.W3))
        )
        
        return tf.reduce_mean(weighted_loss) + l2_loss
    
    def entrenar(self, X_train, y_train, X_val, y_val, epochs=500, batch_size=1024, verbose=True):
        print(f"\n⚡ ENTRENAMIENTO - 3 NODOS CON FOCAL LOSS + BATCH TRAINING\n")
        print(f"Batch size: {batch_size}")
        
        # Mantener numpy para batches
        X_train_np = X_train
        y_train_np = y_train.reshape(-1, 1)
        X_val = tf.constant(X_val, dtype=tf.float32)
        y_val = tf.constant(y_val.reshape(-1, 1), dtype=tf.float32)
        
        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        
        mejor_f1 = 0.0
        sin_mejora = 0
        paciencia = 100  # Aumentado
        
        num_batches = len(X_train_np) // batch_size
        print(f"Batches por época: {num_batches}\n")
        
        for epoch in range(epochs):
            # Mezclar datos cada época
            indices = np.random.permutation(len(X_train_np))
            X_train_shuffled = X_train_np[indices]
            y_train_shuffled = y_train_np[indices]
            
            epoch_loss = []
            epoch_acc = []
            
            # Entrenar por batches
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = tf.constant(X_train_shuffled[start_idx:end_idx], dtype=tf.float32)
                y_batch = tf.constant(y_train_shuffled[start_idx:end_idx], dtype=tf.float32)
                
                with tf.GradientTape() as tape:
                    y_pred, h1, h2, h2_pow, h2_sq, h2_exp, hadam = self.forward(X_batch)
                    perdida = self.calcular_perdida_focal(y_pred, y_batch)
                
                variables = [
                    self.W1, self.b1,
                    self.W2, self.b2,
                    self.W3, self.b3,
                    self.alpha, self.beta, self.gamma, self.delta
                ]
                gradientes = tape.gradient(perdida, variables)
                gradientes_clip = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g for g in gradientes]
                
                optimizer.apply_gradients(zip(gradientes_clip, variables))
                
                epoch_loss.append(perdida.numpy())
                y_pred_batch = (y_pred > 0.5).numpy().astype(int)
                epoch_acc.append(accuracy_score(y_batch.numpy(), y_pred_batch))
            
            # Métricas de época
            perdida_train = np.mean(epoch_loss)
            precision_train = np.mean(epoch_acc)
            
            # Validación
            y_pred_val, h1_val, h2_val, _, _, _, _ = self.forward(X_val)
            perdida_val = self.calcular_perdida_focal(y_pred_val, y_val)
            y_pred_val_class = (y_pred_val > 0.5).numpy().astype(int)
            precision_val = accuracy_score(y_val.numpy(), y_pred_val_class)
            
            from sklearn.metrics import f1_score
            f1_val = f1_score(y_val.numpy(), y_pred_val_class, zero_division=0)
            
            self.historial_perdida.append(perdida_train)
            self.historial_precision.append(precision_train)
            self.historial_perdida_val.append(perdida_val.numpy())
            self.historial_precision_val.append(precision_val)
            self.historial_f1_val.append(f1_val)
            
            self.historial_operaciones.append({
                'alpha': self.alpha.numpy(),
                'beta': self.beta.numpy(),
                'gamma': self.gamma.numpy(),
                'delta': self.delta.numpy(),
                'h1_mean': tf.reduce_mean(h1_val).numpy(),
                'h2_mean': tf.reduce_mean(h2_val).numpy(),
                'h1_std': tf.math.reduce_std(h1_val).numpy(),
                'h2_std': tf.math.reduce_std(h2_val).numpy()
            })
            
            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train → Loss: {perdida_train:.4f} | Acc: {precision_train:.4f}")
                print(f"  Val   → Loss: {perdida_val:.4f} | Acc: {precision_val:.4f} | F1: {f1_val:.4f}")
                print(f"  NODO 3 → α={self.alpha.numpy():.3f} | β={self.beta.numpy():.3f} | γ={self.gamma.numpy():.3f} | δ={self.delta.numpy():.3f}\n")
            
            if f1_val > mejor_f1:
                mejor_f1 = f1_val
                sin_mejora = 0
            else:
                sin_mejora += 1
                if sin_mejora >= paciencia:
                    print(f"✓ Early stopping en época {epoch+1} (Mejor F1: {mejor_f1:.4f})")
                    break
        
        print(f"\n✓ ENTRENAMIENTO COMPLETADO")
        print(f"  Mejor F1-Score: {mejor_f1:.4f}")
        print(f"  Parámetros finales Nodo 3:")
        print(f"    α = {self.alpha.numpy():.4f}")
        print(f"    β = {self.beta.numpy():.4f}")
        print(f"    γ = {self.gamma.numpy():.4f}")
        print(f"    δ = {self.delta.numpy():.4f}")
    
    def predecir(self, X):
        X_tensor = tf.constant(X, dtype=tf.float32)
        y_pred, _, _, _, _, _, _ = self.forward(X_tensor)
        probabilidades = y_pred.numpy()
        predicciones = (probabilidades > 0.5).astype(int)
        return probabilidades, predicciones
    
    def evaluar(self, X_test, y_test):
        probabilidades, predicciones = self.predecir(X_test)
        
        from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
        
        precision = accuracy_score(y_test, predicciones)
        matriz_conf = confusion_matrix(y_test, predicciones)
        f1 = f1_score(y_test, predicciones)
        recall = recall_score(y_test, predicciones)
        precision_score_val = precision_score(y_test, predicciones)
        
        print("\n📊 EVALUACIÓN FINAL - 3 NODOS")
        print(f"Accuracy:   {precision:.4f} ({precision*100:.2f}%)")
        print(f"F1-Score:   {f1:.4f}")
        print(f"Recall:     {recall:.4f} (% fumadores detectados)")
        print(f"Precision:  {precision_score_val:.4f} (precisión fumadores)")
        print("\nMatriz de Confusión:")
        print(matriz_conf)
        print("\n" + classification_report(y_test, predicciones, target_names=['No Fumador', 'Fumador']))
        
        return precision, matriz_conf

# ============================================
# GRÁFICAS DE DISPERSIÓN
# ============================================

def graficar_dispersion_datos(X, y, caracteristicas):
    """Gráfica de dispersión de datos por característica"""
    num_features = min(6, X.shape[1])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    colores = ['blue' if label == 0 else 'red' for label in y]
    
    for i in range(num_features):
        axes[i].scatter(range(len(X)), X[:, i], c=colores, alpha=0.5, s=20, edgecolors='black', linewidth=0.3)
        axes[i].set_xlabel('Índice de Muestra', fontsize=11)
        axes[i].set_ylabel('Valor', fontsize=11)
        axes[i].set_title(f'{caracteristicas[i]}', fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='No Fumador'),
            Patch(facecolor='red', label='Fumador')
        ]
        axes[i].legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    for idx in range(num_features, 6):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('dispersion_datos.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Gráfica guardada: dispersion_datos.png")


def graficar_dispersion_2d(X, y, caracteristicas):
    """Gráfica de dispersión 2D entre pares de características"""
    num_features = min(4, X.shape[1])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    colores = ['blue' if label == 0 else 'red' for label in y]
    
    pair_idx = 0
    for i in range(num_features):
        for j in range(i+1, num_features):
            if pair_idx >= 6:
                break
            
            axes[pair_idx].scatter(X[:, i], X[:, j], c=colores, alpha=0.5, s=30, edgecolors='black', linewidth=0.3)
            axes[pair_idx].set_xlabel(f'{caracteristicas[i]}', fontsize=10)
            axes[pair_idx].set_ylabel(f'{caracteristicas[j]}', fontsize=10)
            axes[pair_idx].set_title(f'{caracteristicas[i]} vs {caracteristicas[j]}', 
                                    fontsize=11, fontweight='bold')
            axes[pair_idx].grid(True, alpha=0.3)
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='blue', label='No Fumador'),
                Patch(facecolor='red', label='Fumador')
            ]
            axes[pair_idx].legend(handles=legend_elements, loc='best', fontsize=8)
            
            pair_idx += 1
            if pair_idx >= 6:
                break
    
    for idx in range(pair_idx, 6):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('dispersion_2d_pares.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Gráfica guardada: dispersion_2d_pares.png")


def graficar_dispersion_residuales(y_test, probabilidades):
    """Gráfica de dispersión de ajustes vs valores reales y residuales"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colores = ['blue' if y == 0 else 'red' for y in y_test]
    ax1.scatter(y_test, probabilidades, alpha=0.6, s=50, c=colores, edgecolors='black', linewidth=0.5)
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Predicción Perfecta')
    ax1.axhline(y=0.5, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Umbral (0.5)')
    ax1.set_xlabel('Valores Reales (0=No Fumador, 1=Fumador)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probabilidades Predichas', fontsize=12, fontweight='bold')
    ax1.set_title('Dispersión: Predicción vs Real', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-0.1, 1.1])
    ax1.set_ylim([-0.1, 1.1])
    
    residuales = y_test - probabilidades.flatten()
    ax2.scatter(probabilidades, residuales, alpha=0.6, s=50, c=colores, edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Residual = 0')
    ax2.set_xlabel('Probabilidades Predichas', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuales (Real - Predicción)', fontsize=12, fontweight='bold')
    ax2.set_title('Gráfica de Residuales', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dispersion_residuales.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Gráfica guardada: dispersion_residuales.png")


def graficar_correlacion_caracteristicas(X, caracteristicas):
    """Matriz de correlación entre características"""
    df_features = pd.DataFrame(X, columns=caracteristicas)
    correlacion = df_features.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlacion, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                annot_kws={'fontsize': 9})
    
    ax.set_title('Matriz de Correlación entre Características', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('correlacion_caracteristicas.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Gráfica guardada: correlacion_caracteristicas.png")

# ============================================
# OTRAS GRÁFICAS
# ============================================

def graficar_distribucion_final(probabilidades, y_test):
    """Distribución de probabilidades"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    probs_no_fumador = probabilidades[y_test == 0]
    probs_fumador = probabilidades[y_test == 1]
    
    ax.hist(probs_no_fumador, bins=60, alpha=0.7, color='blue', 
            label=f'No Fumadores (n={len(probs_no_fumador)})', 
            edgecolor='black', linewidth=0.5)
    ax.hist(probs_fumador, bins=60, alpha=0.7, color='red', 
            label=f'Fumadores (n={len(probs_fumador)})', 
            edgecolor='black', linewidth=0.5)
    ax.axvline(x=0.5, color='green', linestyle='--', linewidth=3, label='Umbral (0.5)')
    
    ax.text(0.02, 0.98, 
            f'No Fumadores:\n  Media: {np.mean(probs_no_fumador):.3f}\n  Std: {np.std(probs_no_fumador):.3f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
    
    ax.text(0.98, 0.98, 
            f'Fumadores:\n  Media: {np.mean(probs_fumador):.3f}\n  Std: {np.std(probs_fumador):.3f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    ax.set_xlabel('Probabilidad de ser Fumador', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=13, fontweight='bold')
    ax.set_title('Distribución - RED DE 3 NODOS', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('distribucion_3_nodos.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Gráfica guardada: distribucion_3_nodos.png")


def graficar_matriz_confusion_3nodos(matriz_conf):
    """Matriz de confusión"""
    fig, ax = plt.subplots(figsize=(9, 7))
    
    sns.heatmap(matriz_conf, annot=True, fmt='d', cmap='RdYlGn_r', 
                xticklabels=['No Fumador', 'Fumador'],
                yticklabels=['No Fumador', 'Fumador'],
                cbar_kws={'label': 'Cantidad'}, annot_kws={'fontsize': 14})
    
    ax.set_xlabel('Predicción', fontsize=13, fontweight='bold')
    ax.set_ylabel('Valor Real', fontsize=13, fontweight='bold')
    ax.set_title('Matriz de Confusión - 3 NODOS', fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('matriz_3_nodos.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Gráfica guardada: matriz_3_nodos.png")


def graficar_operaciones_nodo3(red):
    """Gráfica de las operaciones matemáticas del Nodo 3"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    epochs = range(1, len(red.historial_operaciones) + 1)
    
    alphas = [op['alpha'] for op in red.historial_operaciones]
    betas = [op['beta'] for op in red.historial_operaciones]
    gammas = [op['gamma'] for op in red.historial_operaciones]
    deltas = [op['delta'] for op in red.historial_operaciones]
    h2_means = [op['h2_mean'] for op in red.historial_operaciones]
    
    # Alpha
    axes[0, 0].plot(epochs, alphas, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('α')
    axes[0, 0].set_title('Evolución de α (Nodo 3)', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Beta
    axes[0, 1].plot(epochs, betas, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('β')
    axes[0, 1].set_title('Evolución de β (Nodo 3)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gamma
    axes[1, 0].plot(epochs, gammas, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('γ')
    axes[1, 0].set_title('Evolución de γ (Nodo 3)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Delta
    axes[1, 1].plot(epochs, deltas, 'orange', linewidth=2)
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('δ')
    axes[1, 1].set_title('Evolución de δ (Nodo 3)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Salida Nodo 2
    axes[0, 2].plot(epochs, h2_means, 'm-', linewidth=2)
    axes[0, 2].set_xlabel('Época')
    axes[0, 2].set_ylabel('Salida media h₂')
    axes[0, 2].set_title('Salida promedio del Nodo 2', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Ocultar el último subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('operaciones_nodo3.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Gráfica guardada: operaciones_nodo3.png")


def graficar_entrenamiento(red):
    """Gráfica del progreso de entrenamiento"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(red.historial_perdida) + 1)
    
    # Loss
    axes[0].plot(epochs, red.historial_perdida, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, red.historial_perdida_val, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Época', fontsize=12)
    axes[0].set_ylabel('Pérdida', fontsize=12)
    axes[0].set_title('Pérdida durante Entrenamiento', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy y F1
    axes[1].plot(epochs, red.historial_precision, 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, red.historial_precision_val, 'r-', label='Val Acc', linewidth=2)
    axes[1].plot(epochs, red.historial_f1_val, 'g--', label='Val F1', linewidth=2)
    axes[1].set_xlabel('Época', fontsize=12)
    axes[1].set_ylabel('Métrica', fontsize=12)
    axes[1].set_title('Métricas durante Entrenamiento', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('entrenamiento_3_nodos.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Gráfica guardada: entrenamiento_3_nodos.png")

# ============================================
# FUNCIÓN PRINCIPAL
# ============================================

def main():
    print("\n🧠 RED NEURONAL CON EXACTAMENTE 3 NODOS")
    print("="*60)
    
    db = conectar_mongodb()
    if db is None:
        return
    
    df_procesado = cargar_y_procesar_datos(db)
    if df_procesado is None:
        return
    
    X, y, caracteristicas, class_weights = preparar_dataset(df_procesado)
    if X is None:
        return
    
    # Dividir datos
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"\n📊 Datos: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Gráficas de dispersión de datos (ANTES del entrenamiento)
    print("\n📊 Generando gráficas de dispersión de datos...")
    graficar_dispersion_datos(X, y, caracteristicas)
    graficar_dispersion_2d(X, y, caracteristicas)
    graficar_correlacion_caracteristicas(X, caracteristicas)
    
    # Crear red con 3 NODOS
    red = RedTresNodosMejorada(
        num_entradas=X.shape[1],
        class_weights=class_weights,
        learning_rate=0.001  # AUMENTADO de 0.0005 a 0.001
    )
    
    # Entrenar CON BATCH TRAINING
    red.entrenar(X_train, y_train, X_val, y_val, epochs=500, batch_size=1024, verbose=True)
    
    # Evaluar
    precision, matriz = red.evaluar(X_test, y_test)
    
    # Ejemplos
    print("\n📋 EJEMPLOS DE PREDICCIONES:")
    indices_fumadores = np.where(y_test == 1)[0][:5]
    indices_no_fumadores = np.where(y_test == 0)[0][:5]
    indices = np.concatenate([indices_no_fumadores, indices_fumadores])
    
    probs, preds = red.predecir(X_test[indices])
    
    for idx, i in enumerate(indices):
        print(f"  {idx+1}. Prob: {probs[idx][0]:.3f} → "
              f"{'FUMADOR' if preds[idx][0] == 1 else 'NO FUMADOR'} "
              f"(Real: {'FUMADOR' if y_test[i] == 1 else 'NO FUMADOR'}) "
              f"{'✓' if preds[idx][0] == y_test[i] else '✗'}")
    
    # Gráficas (DESPUÉS del entrenamiento)
    print("\n📊 Generando gráficas de resultados...")
    probs_completas, _ = red.predecir(X_test)
    
    graficar_dispersion_residuales(y_test, probs_completas)
    graficar_distribucion_final(probs_completas, y_test)
    graficar_matriz_confusion_3nodos(matriz)
    graficar_operaciones_nodo3(red)
    graficar_entrenamiento(red)
    
    print("\n" + "="*60)
    print("✅ RED DE 3 NODOS ENTRENADA")
    print(f"NODO 1: {red.num_entradas} → 2 (tanh)")
    print(f"NODO 2: 2 → 1 (relu)")
    print(f"NODO 3: 1 → 1 (α, β, γ, δ + operaciones)")
    print(f"Total: EXACTAMENTE 3 NODOS")
    print(f"Pesos ajustables en Nodo 3:")
    print(f"  α = {red.alpha.numpy():.4f}")
    print(f"  β = {red.beta.numpy():.4f}")
    print(f"  γ = {red.gamma.numpy():.4f}")
    print(f"  δ = {red.delta.numpy():.4f}")
    print("\n📊 GRÁFICAS GENERADAS:")
    print("  1. dispersion_datos.png")
    print("  2. dispersion_2d_pares.png")
    print("  3. correlacion_caracteristicas.png")
    print("  4. dispersion_residuales.png")
    print("  5. distribucion_3_nodos.png")
    print("  6. matriz_3_nodos.png")
    print("  7. operaciones_nodo3.png")
    print("  8. entrenamiento_3_nodos.png")
    print("="*60)

if __name__ == "__main__":
    main()