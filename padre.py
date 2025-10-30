# Clase base o superclase
class Vehiculo:
    def __init__(self, marca, modelo, año):
        self.marca = marca
        self.modelo = modelo
        self.año = año
        self.encendido = False
    
    def encender(self):
        if not self.encendido:
            self.encendido = True
            return f"El {self.marca} {self.modelo} está encendido"
        return "El vehículo ya está encendido"
    
    def apagar(self):
        if self.encendido:
            self.encendido = False
            return f"El {self.marca} {self.modelo} está apagado"
        return "El vehículo ya está apagado"
    
    def info(self):
        return f"{self.marca} {self.modelo} ({self.año})"


# Clase derivada 1: Carro hereda de Vehiculo
class Carro(Vehiculo):
    def __init__(self, marca, modelo, año, num_puertas):
        super().__init__(marca, modelo, año)  # Llama al constructor de la clase padre
        self.num_puertas = num_puertas
    
    def tocar_claxon(self):
        return "¡Beep beep!"
    
    # Sobrescritura de método (override)
    def info(self):
        info_base = super().info()  # Llama al método de la clase padre
        return f"{info_base} - {self.num_puertas} puertas"


# Clase derivada 2: Motocicleta hereda de Vehiculo
class Motocicleta(Vehiculo):
    def __init__(self, marca, modelo, año, tipo):
        super().__init__(marca, modelo, año)
        self.tipo = tipo  # deportiva, crucero, etc.
    
    def hacer_caballito(self):
        if self.encendido:
            return "¡Haciendo un caballito! 🏍️"
        return "Debes encender la moto primero"
    
    def info(self):
        info_base = super().info()
        return f"{info_base} - Tipo: {self.tipo}"


# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancias
    mi_carro = Carro("Toyota", "Corolla", 2022, 4)
    mi_moto = Motocicleta("Yamaha", "R1", 2023, "deportiva")
    
    # Usar métodos heredados
    print(mi_carro.encender())
    print(mi_carro.info())
    print(mi_carro.tocar_claxon())
    
    print("\n")
    
    print(mi_moto.encender())
    print(mi_moto.info())
    print(mi_moto.hacer_caballito())