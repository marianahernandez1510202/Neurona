# Clase base (padre)
class Operacion:
    def __init__(self, num1, num2):
        self.num1 = num1
        self.num2 = num2
    
    def mostrar_numeros(self):
        print(f"Números: {self.num1} y {self.num2}")


# Clases hijas que heredan de Operacion
class Suma(Operacion):
    def calcular(self):
        return self.num1 + self.num2


class Resta(Operacion):
    def calcular(self):
        return self.num1 - self.num25
    


class Multiplicacion(Operacion):
    def calcular(self):
        return self.num1 * self.num2


class Division(Operacion):
    def calcular(self):
        if self.num2 == 0:
            raise ZeroDivisionError("No se puede dividir entre cero")
        return self.num1 / self.num2


class Potencia(Operacion):
    def calcular(self):
        return self.num1 ** self.num2


# Función principal de la calculadora
def calculadora():
    while True:
        try:
            print("\n" + "="*40)
            print("       CALCULADORA CON HERENCIA")
            print("="*40)
            print("1. Suma")
            print("2. Resta")
            print("3. Multiplicación")
            print("4. División")
            print("5. Potencia")
            print("6. Salir")
            print("="*40)
            
            opcion = int(input("Elige una opción: "))
            
            # Switch case para el menú
            match opcion:
                case 1:
                    operacion_nombre = "SUMA"
                    clase_operacion = Suma
                case 2:
                    operacion_nombre = "RESTA"
                    clase_operacion = Resta
                case 3:
                    operacion_nombre = "MULTIPLICACIÓN"
                    clase_operacion = Multiplicacion
                case 4:
                    operacion_nombre = "DIVISIÓN"
                    clase_operacion = Division
                case 5:
                    operacion_nombre = "POTENCIA"
                    clase_operacion = Potencia
                case 6:
                    print("\n¡Gracias por usar la calculadora! 👋")
                    break
                case _:
                    print("\n❌ Opción no válida. Intenta de nuevo.")
                    continue
            
            # Si eligió salir, no pide números
            if opcion == 6:
                break
            
            # Solicitar números
            num1 = float(input(f"\nIngresa el primer número: "))
            num2 = float(input(f"Ingresa el segundo número: "))
            
            # Crear objeto de la clase correspondiente
            operacion = clase_operacion(num1, num2)
            
            # Realizar cálculo
            resultado = operacion.calcular()
            
            # Mostrar resultado
            print(f"\n{'='*40}")
            print(f"   Operación: {operacion_nombre}")
            operacion.mostrar_numeros()
            print(f"   Resultado: {resultado}")
            print(f"{'='*40}")
            
        except ValueError:
            print("\n❌ Error: Debes ingresar un número válido")
        
        except ZeroDivisionError as e:
            print(f"\n❌ Error: {e}")
        
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")
        
        else:
            print("\n✅ Operación completada con éxito")
        
        finally:
            print("\n--- Fin de la operación ---")
            input("Presiona Enter para continuar...")


# Ejecutar la calculadora
if __name__ == "__main__":
    calculadora()