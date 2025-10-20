import kagglehub
import pandas as pd
from pymongo import MongoClient
import numpy as np
import os
import glob

# ============================================
# 1. DESCARGAR DATASET DE KAGGLE
# ============================================

def descargar_dataset_kaggle():
    """
    Descarga el dataset de Kaggle usando kagglehub
    """
    print("📥 Descargando dataset de Kaggle...")
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("sooyoungher/smoking-drinking-dataset")
        print(f"✓ Dataset descargado en: {path}")
        
        # Buscar archivos CSV en la ruta
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        
        if csv_files:
            print(f"\n📄 Archivos CSV encontrados:")
            for i, file in enumerate(csv_files, 1):
                file_name = os.path.basename(file)
                file_size = os.path.getsize(file) / (1024 * 1024)  # MB
                print(f"  {i}. {file_name} ({file_size:.2f} MB)")
            
            return csv_files[0]  # Retornar el primer CSV
        else:
            print("⚠️ No se encontraron archivos CSV en la ruta")
            return None
            
    except Exception as e:
        print(f"✗ Error al descargar: {e}")
        return None

# ============================================
# 2. CONECTAR A MONGODB ATLAS
# ============================================

def conectar_mongodb(uri=None, db_name="salud_db"):
    """
    Conecta a la base de datos MongoDB Atlas
    """
    try:
        # URI de MongoDB Atlas
        if uri is None:
            uri = "mongodb+srv://2022371082_db_user:marianahernandezdimas15102004@cluster0.gtmppy1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        
        print(f"🔗 Conectando a MongoDB Atlas...")
        client = MongoClient(uri)
        
        # Verificar conexión
        client.admin.command('ping')
        
        # Seleccionar/crear base de datos
        db = client[db_name]
        
        print(f"✓ Conexión exitosa a MongoDB Atlas")
        print(f"✓ Base de datos: {db_name}")
        
        # Listar bases de datos existentes
        print(f"\n📚 Bases de datos disponibles en tu cluster:")
        for db_info in client.list_databases():
            print(f"   • {db_info['name']}")
        
        return client, db
        
    except Exception as e:
        print(f"✗ Error al conectar a MongoDB Atlas: {e}")
        print(f"\n💡 Verifica:")
        print(f"   • Tu conexión a internet")
        print(f"   • Las credenciales de MongoDB Atlas")
        print(f"   • Las reglas de firewall (IP whitelist)")
        return None, None

# ============================================
# 3. CARGAR CSV A MONGODB
# ============================================

def cargar_csv_a_mongodb(csv_path, db, coleccion_nombre="encuestas_fumadores", 
                         batch_size=1000, max_registros=None):
    """
    Carga el archivo CSV a MongoDB en lotes (batches)
    
    Args:
        csv_path: Ruta al archivo CSV
        db: Instancia de la base de datos MongoDB
        coleccion_nombre: Nombre de la colección
        batch_size: Número de documentos por lote
        max_registros: Límite de registros a insertar (None = todos)
    """
    try:
        print(f"\n📖 Leyendo archivo CSV...")
        
        # Leer CSV (si es muy grande, usar chunks)
        if max_registros:
            df = pd.read_csv(csv_path, nrows=max_registros)
            print(f"   Limitando a {max_registros} registros")
        else:
            df = pd.read_csv(csv_path)
        
        print(f"✓ Archivo leído: {len(df)} registros, {len(df.columns)} columnas")
        
        # Mostrar información del dataset
        print(f"\n📊 Información del dataset:")
        print(f"   Columnas: {list(df.columns[:10])}...")
        print(f"   Forma: {df.shape}")
        print(f"   Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Limpiar valores NaN (MongoDB no acepta NaN de NumPy)
        print(f"\n🧹 Limpiando valores nulos...")
        df = df.replace({np.nan: None})
        
        # Convertir a lista de diccionarios
        print(f"\n📝 Convirtiendo a formato MongoDB...")
        datos = df.to_dict('records')
        
        # Obtener o crear colección
        collection = db[coleccion_nombre]
        
        # Opcional: Limpiar colección existente
        respuesta = input(f"\n⚠️  ¿Deseas limpiar la colección '{coleccion_nombre}' antes de insertar? (s/n): ")
        if respuesta.lower() == 's':
            collection.delete_many({})
            print(f"✓ Colección limpiada")
        
        # Insertar en lotes
        print(f"\n💾 Insertando datos en MongoDB en lotes de {batch_size}...")
        total_insertados = 0
        
        for i in range(0, len(datos), batch_size):
            batch = datos[i:i + batch_size]
            resultado = collection.insert_many(batch)
            total_insertados += len(resultado.inserted_ids)
            
            # Mostrar progreso
            progreso = (total_insertados / len(datos)) * 100
            print(f"   Progreso: {total_insertados}/{len(datos)} ({progreso:.1f}%)", end='\r')
        
        print(f"\n✓ {total_insertados} documentos insertados exitosamente")
        
        # Crear índices para búsquedas más rápidas
        print(f"\n🔍 Creando índices...")
        collection.create_index("age")
        collection.create_index("sex")
        collection.create_index("smoking")
        print(f"✓ Índices creados")
        
        # Estadísticas de la colección
        print(f"\n📈 Estadísticas de la colección:")
        print(f"   Total documentos: {collection.count_documents({})}")
        print(f"   Tamaño estimado: {collection.estimated_document_count()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error al cargar datos: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================
# 4. EXPLORAR DATOS EN MONGODB
# ============================================

def explorar_coleccion(db, coleccion_nombre="encuestas_fumadores"):
    """
    Muestra información básica de la colección
    """
    try:
        collection = db[coleccion_nombre]
        
        print(f"\n" + "="*60)
        print(f"EXPLORACIÓN DE LA COLECCIÓN: {coleccion_nombre}")
        print("="*60)
        
        # Contar documentos
        total = collection.count_documents({})
        print(f"\n📊 Total de documentos: {total}")
        
        # Mostrar un documento de ejemplo
        print(f"\n📄 Documento de ejemplo:")
        ejemplo = collection.find_one()
        if ejemplo:
            # Mostrar solo algunos campos
            campos_mostrar = list(ejemplo.keys())[:15]
            for campo in campos_mostrar:
                print(f"   {campo}: {ejemplo[campo]}")
            
            if len(ejemplo.keys()) > 15:
                print(f"   ... y {len(ejemplo.keys()) - 15} campos más")
        
        # Estadísticas básicas
        print(f"\n📈 Estadísticas de fumadores:")
        
        # Contar fumadores (ajustar el nombre del campo según tu dataset)
        # El dataset de Kaggle usa 'smoking' como campo
        pipeline = [
            {"$group": {
                "_id": "$smoking",
                "count": {"$sum": 1}
            }}
        ]
        
        resultados = list(collection.aggregate(pipeline))
        for resultado in resultados:
            estado = "Fumador" if resultado['_id'] == 1 else "No fumador"
            porcentaje = (resultado['count'] / total) * 100
            print(f"   {estado}: {resultado['count']} ({porcentaje:.2f}%)")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"✗ Error al explorar colección: {e}")

# ============================================
# 5. CONVERTIR A FORMATO BOOLEANO
# ============================================

def procesar_para_neurona(db, coleccion_origen="encuestas_fumadores", 
                         coleccion_destino="encuestas_procesadas"):
    """
    Procesa los datos y crea una colección con variables booleanas
    para la neurona artificial
    """
    try:
        print(f"\n🔄 Procesando datos para la neurona...")
        
        collection_origen = db[coleccion_origen]
        collection_destino = db[coleccion_destino]
        
        # Limpiar colección destino
        collection_destino.delete_many({})
        
        # Obtener todos los documentos
        documentos = list(collection_origen.find())
        df = pd.DataFrame(documentos)
        
        print(f"✓ {len(df)} documentos obtenidos")
        
        # Crear DataFrame procesado con variables booleanas
        df_procesado = pd.DataFrame()
        
        # Variables booleanas básicas (ajustar según las columnas reales del dataset)
        # Este dataset tiene muchas columnas, aquí procesamos las más relevantes
        
        # 1. Fumador (ya es binario en el dataset)
        if 'smoking' in df.columns:
            df_procesado['fuma'] = df['smoking']
        
        # 2. Sexo (convertir a binario: Male=1, Female=0)
        if 'sex' in df.columns:
            df_procesado['sexo_masculino'] = df['sex'].apply(
                lambda x: 1 if str(x).lower() == 'male' else 0
            )
        
        # 3. Edad normalizada
        if 'age' in df.columns:
            edad_min = df['age'].min()
            edad_max = df['age'].max()
            df_procesado['edad_normalizada'] = (df['age'] - edad_min) / (edad_max - edad_min)
            df_procesado['edad'] = df['age']
        
        # 4. Altura normalizada
        if 'height(cm)' in df.columns:
            altura_min = df['height(cm)'].min()
            altura_max = df['height(cm)'].max()
            df_procesado['altura_normalizada'] = (df['height(cm)'] - altura_min) / (altura_max - altura_min)
        
        # 5. Peso normalizado
        if 'weight(kg)' in df.columns:
            peso_min = df['weight(kg)'].min()
            peso_max = df['weight(kg)'].max()
            df_procesado['peso_normalizado'] = (df['weight(kg)'] - peso_min) / (peso_max - peso_min)
        
        # 6. IMC (Body Mass Index) - si no existe, calcular
        if 'height(cm)' in df.columns and 'weight(kg)' in df.columns:
            altura_m = df['height(cm)'] / 100
            df_procesado['imc'] = df['weight(kg)'] / (altura_m ** 2)
            df_procesado['tiene_obesidad'] = (df_procesado['imc'] > 30).astype(int)
            df_procesado['tiene_sobrepeso'] = (df_procesado['imc'] > 25).astype(int)
        
        # 7. Presión arterial alta (si existe systolic o diastolic)
        if 'systolic' in df.columns:
            df_procesado['presion_alta'] = (df['systolic'] > 140).astype(int)
        
        # 8. Niveles de colesterol (si existen)
        if 'tot_chole' in df.columns:
            df_procesado['colesterol_alto'] = (df['tot_chole'] > 200).astype(int)
        
        # 9. Nivel de glucosa
        if 'fasting blood sugar' in df.columns:
            df_procesado['glucosa_alta'] = (df['fasting blood sugar'] > 100).astype(int)
        
        # 10. Consumo de alcohol (si existe)
        if 'DRK_YN' in df.columns:
            df_procesado['bebe_alcohol'] = df['DRK_YN']
        
        # Guardar en nueva colección
        datos_procesados = df_procesado.to_dict('records')
        
        if datos_procesados:
            collection_destino.insert_many(datos_procesados)
            print(f"✓ {len(datos_procesados)} documentos procesados")
            print(f"\n📋 Variables booleanas creadas:")
            for col in df_procesado.columns:
                print(f"   • {col}")
        
        return df_procesado
        
    except Exception as e:
        print(f"✗ Error al procesar datos: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================
# 6. FUNCIÓN PRINCIPAL
# ============================================

def main():
    """
    Función principal que ejecuta todo el flujo
    """
    print("\n" + "="*60)
    print("KAGGLE → MONGODB → NEURONA ARTIFICIAL")
    print("="*60 + "\n")
    
    # 1. Descargar dataset
    csv_path = descargar_dataset_kaggle()
    
    if csv_path is None:
        print("\n✗ No se pudo descargar el dataset")
        return
    
    # 2. Conectar a MongoDB
    client, db = conectar_mongodb()
    
    if db is None:
        print("\n✗ No se pudo conectar a MongoDB")
        return
    
    # 3. Preguntar si quiere limitar registros (dataset es muy grande)
    print(f"\n⚠️  NOTA: Este dataset tiene ~991,000 registros.")
    print(f"   Para pruebas, se recomienda usar una muestra más pequeña.")
    
    opcion = input(f"\n¿Cuántos registros deseas cargar? (Enter para todos, o un número): ")
    
    max_registros = None
    if opcion.strip().isdigit():
        max_registros = int(opcion.strip())
    
    # 4. Cargar CSV a MongoDB
    exito = cargar_csv_a_mongodb(
        csv_path, 
        db, 
        coleccion_nombre="encuestas_fumadores",
        batch_size=1000,
        max_registros=max_registros
    )
    
    if not exito:
        print("\n✗ Error al cargar datos a MongoDB")
        client.close()
        return
    
    # 5. Explorar colección
    explorar_coleccion(db, "encuestas_fumadores")
    
    # 6. Procesar para la neurona
    procesar = input(f"\n¿Deseas procesar los datos para la neurona artificial? (s/n): ")
    
    if procesar.lower() == 's':
        df_procesado = procesar_para_neurona(db)
        
        if df_procesado is not None:
            print(f"\n✅ Datos procesados y listos para la neurona")
            explorar_coleccion(db, "encuestas_procesadas")
    
    # 7. Cerrar conexión
    client.close()
    print(f"\n✅ Proceso completado exitosamente")
    print(f"\n💡 Ahora puedes usar la colección 'encuestas_procesadas' con tu neurona")

# ============================================
# EJECUTAR
# ============================================

if __name__ == "__main__":
    main()