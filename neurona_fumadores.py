import tensorflow as tf
import numpy as np
from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================
# 1. CONEXIÓN A MONGODB ATLAS
# ============================================

def conectar_mongodb(uri=None, db_name="salud_db"):
    """
    Conecta a la base de datos MongoDB Atlas
    """
    try:
        if uri is None:
            uri = "mongodb+srv://2022371082_db_user:marianahernandezdimas15102004@cluster0.gtmppy1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        
        client = MongoClient(uri)
        client.admin.command('ping')
        
        db = client[db_name]
        print(f"✓ Conexión exitosa a MongoDB Atlas - Base de datos: {db_name}")
        return db
    except Exception as e:
        print(f"✗ Error al conectar: {e}")
        return None

# ============================================
# 2. CARGAR Y PROCESAR DATOS DESDE MONGODB
# ============================================

def cargar_y_procesar_datos(db):
    """
    Carga datos desde encuestas_fumadores y los procesa
    """
    try:
        print("\n📊 Cargando datos desde 'encuestas_fumadores'...")
        
        # Cargar TODOS los datos de la colección original
        print("  ️  Cargando todos los registros (esto puede tardar un momento)...")
        encuestas = db.encuestas_fumadores.find()
        datos = list(encuestas)
        
        if not datos:
            print(" No hay datos en 'encuestas_fumadores'")
            return None
        
        df = pd.DataFrame(datos)
        print(f"✓ Datos cargados: {len(df)} registros")
        print(f"  Columnas disponibles: {len(df.columns)}")
        
        # ============================================
        # PROCESAMIENTO: Convertir a variables booleanas
        # ============================================
        print(f"\n🔄 Procesando variables...")
        
        df_procesado = pd.DataFrame()
        
        # 1. VARIABLE OBJETIVO: Fumador (desde SMK_stat_type_cd)
        # 1 = nunca fumó, 2 = ex-fumador, 3 = fumador actual
        if 'SMK_stat_type_cd' in df.columns:
            df_procesado['fuma'] = (df['SMK_stat_type_cd'] == 3).astype(int)
            fumadores = df_procesado['fuma'].sum()
            print(f"  ✓ 'fuma': Creada desde SMK_stat_type_cd ({fumadores} fumadores)")
        else:
            print(f"  ❌ No se encontró SMK_stat_type_cd")
            return None
        
        # 2. Sexo (Male=1, Female=0)
        if 'sex' in df.columns:
            df_procesado['sexo_masculino'] = (df['sex'] == 'Male').astype(int)
            print(f"  ✓ 'sexo_masculino': Creada")
        
        # 3. Edad normalizada
        if 'age' in df.columns:
            edad_min, edad_max = df['age'].min(), df['age'].max()
            df_procesado['edad_normalizada'] = (df['age'] - edad_min) / (edad_max - edad_min)
            print(f"  ✓ 'edad_normalizada': Creada")
        
        # 4. Presión alta (SBP > 140)
        if 'SBP' in df.columns:
            df_procesado['presion_alta'] = (df['SBP'] > 140).astype(int)
            print(f"  ✓ 'presion_alta': Creada")
        
        # 5. Colesterol alto (tot_chole > 200)
        if 'tot_chole' in df.columns:
            df_procesado['colesterol_alto'] = (df['tot_chole'] > 200).astype(int)
            print(f"  ✓ 'colesterol_alto': Creada")
        
        # 6. Triglicéridos altos (> 150)
        if 'triglyceride' in df.columns:
            df_procesado['trigliceridos_altos'] = (df['triglyceride'] > 150).astype(int)
            print(f"  ✓ 'trigliceridos_altos': Creada")
        
        # 7. Glucosa alta (BLDS > 100)
        if 'BLDS' in df.columns:
            df_procesado['glucosa_alta'] = (df['BLDS'] > 100).astype(int)
            print(f"  ✓ 'glucosa_alta': Creada")
        
        # 8. Consumo de alcohol (DRK_YN = 'Y' → 1, 'N' → 0)
        if 'DRK_YN' in df.columns:
            df_procesado['bebe_alcohol'] = (df['DRK_YN'] == 'Y').astype(int)
            print(f"  ✓ 'bebe_alcohol': Creada y convertida a 1/0")
        
        # 9. IMC y obesidad
        if 'height' in df.columns and 'weight' in df.columns:
            # Altura en cm, peso en kg
            altura_m = df['height'] / 100
            imc = df['weight'] / (altura_m ** 2)
            df_procesado['tiene_obesidad'] = (imc > 30).astype(int)
            df_procesado['tiene_sobrepeso'] = (imc > 25).astype(int)
            print(f"  ✓ 'tiene_obesidad' y 'tiene_sobrepeso': Creadas")
        
        # 10. Cintura alta (waistline > 90 hombres, > 85 mujeres)
        if 'waistline' in df.columns and 'sex' in df.columns:
            cintura_alta = []
            for idx, row in df.iterrows():
                if row['sex'] == 'Male':
                    cintura_alta.append(1 if row['waistline'] > 90 else 0)
                else:
                    cintura_alta.append(1 if row['waistline'] > 85 else 0)
            df_procesado['cintura_alta'] = cintura_alta
            print(f"  ✓ 'cintura_alta': Creada")
        
        # Eliminar filas con valores nulos
        df_procesado = df_procesado.dropna()
        
        print(f"\n✓ Procesamiento completado:")
        print(f"  • {len(df_procesado)} registros válidos")
        print(f"  • {len(df_procesado.columns)} variables")
        
        return df_procesado
    
    except Exception as e:
        print(f"✗ Error al cargar/procesar datos: {e}")
        import traceback
        traceback.print_exc()
        return None

def preparar_dataset(df_procesado):
    """
    Prepara X (características) y y (etiquetas) para entrenamiento
    """
    # Todas las columnas excepto 'fuma'
    caracteristicas = [col for col in df_procesado.columns if col != 'fuma']
    
    if not caracteristicas:
        print(" No se encontraron características")
        return None, None, None
    
    X = df_procesado[caracteristicas].values.astype(np.float32)
    y = df_procesado['fuma'].values.astype(np.float32)
    
    print(f"\n✓ Dataset preparado:")
    print(f"  • {X.shape[0]} muestras")
    print(f"  • {X.shape[1]} características: {caracteristicas}")
    print(f"  • Fumadores: {int(y.sum())} ({(y.sum()/len(y))*100:.1f}%)")
    print(f"  • No fumadores: {int(len(y) - y.sum())} ({((len(y)-y.sum())/len(y))*100:.1f}%)")
    
    return X, y, caracteristicas

# ============================================
# 3. NEURONA ARTIFICIAL CON ENTRENAMIENTO
# ============================================

class NeuronaEntrenada:
    """
    Neurona artificial que puede ser ENTRENADA
    """
    
    def __init__(self, num_entradas, learning_rate=0.01):
        self.num_entradas = num_entradas
        self.learning_rate = learning_rate
        
        # Inicializar pesos y sesgo
        self.pesos = tf.Variable(
            tf.random.normal([num_entradas, 1], mean=0.0, stddev=0.1),
            name='pesos',
            trainable=True
        )
        self.sesgo = tf.Variable(
            tf.zeros([1]),
            name='sesgo',
            trainable=True
        )
        
        # Historial
        self.historial_perdida = []
        self.historial_precision = []
        
        print(f"\n✓ Neurona creada:")
        print(f"  • {num_entradas} entradas")
        print(f"  • Tasa de aprendizaje: {learning_rate}")
        print(f"  • Pesos: {self.pesos.shape}")
    
    def forward(self, X):
        """Propagación hacia adelante"""
        z = tf.matmul(X, self.pesos) + self.sesgo
        return tf.sigmoid(z)
    
    def calcular_perdida(self, y_pred, y_real):
        """Función de pérdida (Binary Cross-Entropy)"""
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        perdida = -tf.reduce_mean(
            y_real * tf.math.log(y_pred) + 
            (1 - y_real) * tf.math.log(1 - y_pred)
        )
        return perdida
    
    def entrenar(self, X_train, y_train, X_val, y_val, epochs=100, verbose=True):
        """AQUÍ ES DONDE LA NEURONA APRENDE"""
        
        print(f"\n{'='*60}")
        print("🎓 INICIANDO ENTRENAMIENTO")
        print(f"{'='*60}\n")
        
        # Convertir a tensores
        X_train = tf.constant(X_train, dtype=tf.float32)
        y_train = tf.constant(y_train.reshape(-1, 1), dtype=tf.float32)
        X_val = tf.constant(X_val, dtype=tf.float32)
        y_val = tf.constant(y_val.reshape(-1, 1), dtype=tf.float32)
        
        # Optimizador
        optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)
        
        mejor_perdida_val = float('inf')
        sin_mejora = 0
        paciencia = 10
        
        for epoch in range(epochs):
            # PASO 1: Forward pass
            with tf.GradientTape() as tape:
                y_pred = self.forward(X_train)
                perdida = self.calcular_perdida(y_pred, y_train)
            
            # PASO 2: Backward pass (calcular gradientes)
            gradientes = tape.gradient(perdida, [self.pesos, self.sesgo])
            
            # PASO 3: Actualizar pesos (AQUÍ APRENDE)
            optimizer.apply_gradients(zip(gradientes, [self.pesos, self.sesgo]))
            
            # Calcular precisión
            y_pred_train = (y_pred > 0.5).numpy().astype(int)
            precision_train = accuracy_score(y_train.numpy(), y_pred_train)
            
            # Evaluar en validación
            y_pred_val = self.forward(X_val)
            perdida_val = self.calcular_perdida(y_pred_val, y_val)
            y_pred_val_class = (y_pred_val > 0.5).numpy().astype(int)
            precision_val = accuracy_score(y_val.numpy(), y_pred_val_class)
            
            # Guardar historial
            self.historial_perdida.append(perdida.numpy())
            self.historial_precision.append(precision_train)
            
            # Mostrar progreso cada 10 epochs
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Loss Train: {perdida:.4f} | Acc Train: {precision_train:.4f}")
                print(f"  Loss Val:   {perdida_val:.4f} | Acc Val:   {precision_val:.4f}")
                print()
            
            # Early stopping
            if perdida_val < mejor_perdida_val:
                mejor_perdida_val = perdida_val
                sin_mejora = 0
            else:
                sin_mejora += 1
                if sin_mejora >= paciencia:
                    print(f"️ Early stopping en epoch {epoch+1}")
                    break
        
        print(f"{'='*60}")
        print("✅ ENTRENAMIENTO COMPLETADO")
        print(f"{'='*60}\n")
        print(f"📊 Resultados finales:")
        print(f"  • Pérdida final (train): {perdida:.4f}")
        print(f"  • Precisión final (train): {precision_train:.4f}")
        print(f"  • Pérdida final (val): {perdida_val:.4f}")
        print(f"  • Precisión final (val): {precision_val:.4f}")
    
    def predecir(self, X):
        """Hace predicciones"""
        X_tensor = tf.constant(X, dtype=tf.float32)
        probabilidades = self.forward(X_tensor)
        predicciones = (probabilidades > 0.5).numpy().astype(int)
        return probabilidades.numpy(), predicciones
    
    def evaluar(self, X_test, y_test):
        """Evalúa el rendimiento"""
        probabilidades, predicciones = self.predecir(X_test)
        
        precision = accuracy_score(y_test, predicciones)
        matriz_conf = confusion_matrix(y_test, predicciones)
        reporte = classification_report(y_test, predicciones, 
                                       target_names=['No Fumador', 'Fumador'])
        
        print(f"\n{'='*60}")
        print("📈 EVALUACIÓN EN TEST")
        print(f"{'='*60}\n")
        print(f"Precisión (Accuracy): {precision:.4f} ({precision*100:.2f}%)\n")
        
        print("Matriz de Confusión:")
        print(f"                 Predicho No | Predicho Sí")
        print(f"Real No Fuma:    {matriz_conf[0][0]:>11} | {matriz_conf[0][1]:>11}")
        print(f"Real Sí Fuma:    {matriz_conf[1][0]:>11} | {matriz_conf[1][1]:>11}")
        print()
        
        print("Reporte de Clasificación:")
        print(reporte)
        
        return precision, matriz_conf

# ============================================
# 4. FUNCIÓN PRINCIPAL
# ============================================

def main():
    """
    Flujo completo: Cargar → Entrenar → Evaluar
    """
    print("\n" + "="*60)
    print("NEURONA ARTIFICIAL CON ENTRENAMIENTO")
    print("Análisis de Fumadores - MongoDB Atlas")
    print("="*60)
    
    # 1. Conectar a MongoDB
    db = conectar_mongodb()
    if db is None:
        return
    
    # 2. Cargar y procesar datos
    df_procesado = cargar_y_procesar_datos(db)
    if df_procesado is None:
        return
    
    # 3. Preparar dataset
    X, y, caracteristicas = preparar_dataset(df_procesado)
    if X is None:
        return
    
    # 4. Dividir en train, validation y test (60%, 20%, 20%)
    print(f"\n📊 Dividiendo datos en train/val/test...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"  • Train: {len(X_train)} muestras ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  • Val:   {len(X_val)} muestras ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  • Test:  {len(X_test)} muestras ({len(X_test)/len(X)*100:.1f}%)")
    
    # 5. Crear neurona
    neurona = NeuronaEntrenada(
        num_entradas=X.shape[1],
        learning_rate=0.1
    )
    
    # 6. ENTRENAR (¡AQUÍ ES DONDE APRENDE!)
    neurona.entrenar(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        verbose=True
    )
    
    # 7. Evaluar en conjunto de test
    precision, matriz = neurona.evaluar(X_test, y_test)
    
    # 8. Mostrar algunos ejemplos
    print(f"\n{'='*60}")
    print("🔍 EJEMPLOS DE PREDICCIONES")
    print(f"{'='*60}\n")
    
    probabilidades, predicciones = neurona.predecir(X_test[:10])
    
    for i in range(10):
        print(f"Ejemplo {i+1}:")
        print(f"  Probabilidad: {probabilidades[i][0]:.4f}")
        print(f"  Predicción: {'🚬 FUMADOR' if predicciones[i][0] == 1 else '🚭 NO FUMADOR'}")
        print(f"  Real: {'🚬 FUMADOR' if y_test[i] == 1 else '🚭 NO FUMADOR'}")
        print(f"  {'✓ CORRECTO' if predicciones[i][0] == y_test[i] else '✗ INCORRECTO'}")
        print()
    
    print("="*60)
    print("\n✅ ¡LA NEURONA HA APRENDIDO!")
    print(f"\n📝 Resumen:")
    print(f"  • La neurona ajustó sus pesos durante el entrenamiento")
    print(f"  • Precisión alcanzada: {precision*100:.2f}%")
    print(f"  • Características usadas: {len(caracteristicas)}")
    print(f"  • Total de parámetros: {neurona.num_entradas + 1}")
    print(f"\n💾 Datos: MongoDB Atlas → salud_db → encuestas_fumadores")

if __name__ == "__main__":
    main()