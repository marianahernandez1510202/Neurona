import numpy as np
import matplotlib.pyplot as plt

# Función sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Crear valores de z de -10 a 10
z = np.linspace(-10, 10, 1000)
y = sigmoid(z)

# Crear la gráfica
plt.figure(figsize=(12, 8))

# Gráfica principal
plt.plot(z, y, 'b-', linewidth=3, label='σ(z) = 1/(1+e⁻ᶻ)')

# Línea horizontal en y=0.5 (umbral de decisión)
plt.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Umbral de decisión (0.5)')

# Línea vertical en z=0
plt.axvline(x=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

# Líneas horizontales en y=0 y y=1
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.axhline(y=1, color='black', linestyle='-', linewidth=0.5)

# Marcar puntos importantes
puntos_z = [-5, -2, 0, 2, 5]
puntos_y = sigmoid(np.array(puntos_z))

for z_val, y_val in zip(puntos_z, puntos_y):
    plt.plot(z_val, y_val, 'ro', markersize=8)
    plt.annotate(f'({z_val}, {y_val:.3f})', 
                xy=(z_val, y_val), 
                xytext=(z_val+0.5, y_val+0.08),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Etiquetas y título
plt.xlabel('z (Suma Ponderada)', fontsize=14, fontweight='bold')
plt.ylabel('σ(z) - Probabilidad', fontsize=14, fontweight='bold')
plt.title('Función de Activación Sigmoid\nUsada en la Neurona Artificial', 
          fontsize=16, fontweight='bold', pad=20)

# Leyenda
plt.legend(fontsize=12, loc='upper left')

# Grid
plt.grid(True, alpha=0.3, linestyle='--')

# Anotaciones explicativas
plt.text(-8, 0.85, 'z muy negativo\n→ σ(z) ≈ 0\n→ NO FUMADOR', 
         fontsize=11, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.text(5, 0.15, 'z muy positivo\n→ σ(z) ≈ 1\n→ FUMADOR', 
         fontsize=11, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

plt.text(-1, 0.6, 'z = 0\nσ(z) = 0.5\nINDECISO', 
         fontsize=11, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Ajustar límites
plt.xlim(-10, 10)
plt.ylim(-0.1, 1.1)

# Ajustar diseño
plt.tight_layout()

# Guardar la gráfica
plt.savefig('funcion_sigmoid.png', dpi=300, bbox_inches='tight')
print("✅ Gráfica guardada como 'funcion_sigmoid.png'")

# Mostrar la gráfica
plt.show()

# ============================================
# GRÁFICA ADICIONAL: Comparación con ejemplos
# ============================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Función sigmoid con ejemplos
ax1.plot(z, y, 'b-', linewidth=3)
ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=2)
ax1.axvline(x=0, color='gray', linestyle=':', linewidth=1.5)

# Ejemplos de predicción
ejemplos = [
    (-3, 'Perfil saludable\nz = -3', 'green'),
    (0, 'Neutral\nz = 0', 'orange'),
    (3, 'Múltiples factores\nz = 3', 'red')
]

for z_ejemplo, texto, color in ejemplos:
    y_ejemplo = sigmoid(z_ejemplo)
    ax1.plot(z_ejemplo, y_ejemplo, 'o', markersize=15, color=color)
    ax1.annotate(texto, xy=(z_ejemplo, y_ejemplo), 
                xytext=(z_ejemplo, y_ejemplo + 0.15),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

ax1.set_xlabel('z (Suma Ponderada)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Probabilidad', fontsize=12, fontweight='bold')
ax1.set_title('Función Sigmoid con Ejemplos', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-10, 10)
ax1.set_ylim(-0.1, 1.1)

# Subplot 2: Tabla de valores
ax2.axis('off')

# Crear tabla de valores
z_vals = np.array([-5, -3, -2, -1, 0, 1, 2, 3, 5])
sigmoid_vals = sigmoid(z_vals)
decision = ['NO FUMA' if s < 0.5 else 'FUMA' for s in sigmoid_vals]

tabla_data = []
for z_val, sig_val, dec in zip(z_vals, sigmoid_vals, decision):
    tabla_data.append([f'{z_val:+.0f}', f'{sig_val:.4f}', f'{sig_val*100:.2f}%', dec])

tabla = ax2.table(cellText=tabla_data,
                 colLabels=['z', 'σ(z)', 'Probabilidad %', 'Decisión'],
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.15, 0.2, 0.25, 0.25])

tabla.auto_set_font_size(False)
tabla.set_fontsize(11)
tabla.scale(1, 2.5)

# Colorear encabezados
for i in range(4):
    tabla[(0, i)].set_facecolor('#4472C4')
    tabla[(0, i)].set_text_props(weight='bold', color='white')

# Colorear filas según decisión
for i in range(1, len(tabla_data) + 1):
    if tabla_data[i-1][3] == 'NO FUMA':
        color = '#90EE90'  # Verde claro
    else:
        color = '#FFB6C6'  # Rosa claro
    for j in range(4):
        tabla[(i, j)].set_facecolor(color)

ax2.set_title('Tabla de Valores de la Función Sigmoid', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('sigmoid_completa.png', dpi=300, bbox_inches='tight')
print("✅ Gráfica completa guardada como 'sigmoid_completa.png'")

plt.show()

print("\n" + "="*60)
print("📊 INFORMACIÓN DE LA FUNCIÓN SIGMOID")
print("="*60)
print("\n📐 Fórmula: σ(z) = 1 / (1 + e^(-z))")
print("\n📈 Características:")
print("  • Rango: (0, 1) - Siempre da valores entre 0 y 1")
print("  • Umbral: 0.5 - Valores > 0.5 → Fumador, < 0.5 → No fumador")
print("  • Suave: No tiene saltos bruscos")
print("  • Derivable: Permite calcular gradientes para entrenar")
print("\n💡 Interpretación:")
print("  • σ(z) = 0.9 → 90% de confianza que es FUMADOR")
print("  • σ(z) = 0.5 → 50% - INDECISO")
print("  • σ(z) = 0.1 → 10% - Muy seguro que NO FUMA")
print("\n✅ Gráficas generadas exitosamente!")