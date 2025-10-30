class Vehiculo:
    def __init__ (self, marca, modelo, año):
        self.marca = marca
        self.modelo = modelo
        self.año = año
        self.encendido = False

    def encender(self):
        if not self.encendido:
            self.encendido = true
            return f"el {self.marca} {self.modelo} esta encendido" 
        return "el vehiculo ya esta encendido"    
      
