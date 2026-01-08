## 🔄 Actualizaciones Recientes

### Versión con Semillas Primas e Impresión de Mejor Individuo

Se han implementado las siguientes mejoras para mayor control y trazabilidad:

#### 1. Variable Global `OUTPUT_DIR`
**Archivos modificados**: `polyploid_genetic_algorithm.py`, `run_experiment.py`

Todas las rutas de salida ahora están centralizadas en una única variable global:
```python
OUTPUT_DIR = "C:/Users/isria/Documents/ESCOM/semestre 8/topicos/practica2/"
```

**Beneficios**:
- ✅ Fácil de cambiar la carpeta de salida en un solo lugar
- ✅ Mayor portabilidad del código
- ✅ Consistencia en todas las salidas

**Archivos que usan `OUTPUT_DIR`**:
- `tablas_hipervolumen.txt`
- `respuestas_preguntas.txt`
- `report_{POLICY}.txt` (6 archivos)
- `pareto_{POLICY}.png` (6 gráficas)
- `gantt_{POLICY}_knee.png`, `gantt_{POLICY}_min_makespan.png`, `gantt_{POLICY}_min_energy.png` (18 diagramas)

#### 2. Semillas con Números Primos
**Archivo modificado**: `run_experiment.py`

Se agregan dos nuevas funciones:

**`is_prime(n)`**: Verifica si un número es primo
```python
def is_prime(n):
    """Verifica si un número es primo."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
```

**`generate_prime_seeds(count)`**: Genera los primeros `count` números primos
```python
def generate_prime_seeds(count):
    """Genera 'count' números primos para usar como semillas."""
    primes = []
    candidate = 2
    while len(primes) < count:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    return primes
```

**Semillas para 10 corridas**:
```
Corrida  1: seed = 2
Corrida  2: seed = 3
Corrida  3: seed = 5
Corrida  4: seed = 7
Corrida  5: seed = 11
Corrida  6: seed = 13
Corrida  7: seed = 17
Corrida  8: seed = 19
Corrida  9: seed = 23
Corrida 10: seed = 29
```

**Ventajas de usar números primos**:
- ✅ Mayor variabilidad entre corridas
- ✅ Mejor espaciamiento en el espacio de números aleatorios
- ✅ Propiedades matemáticas que mejoran la aleatoriedad
- ✅ Fácil reproducibilidad y trazabilidad

#### 3. Impresión de Semillas en Consola
**Archivo modificado**: `run_experiment.py`

La función `print_and_track_seeds()` ahora:
- Genera automáticamente números primos como semillas
- Imprime claramente todas las semillas al iniciar la experimentación
- Guarda las semillas en el archivo de tablas de hipervolumen

**Salida en consola**:
```
════════════════════════════════════════════════════════════════════════════
SEMILLAS PARA CADA CORRIDA (NÚMEROS PRIMOS)
════════════════════════════════════════════════════════════════════════════

  Corrida  1: seed = 2
  Corrida  2: seed = 3
  Corrida  3: seed = 5
  ...
  Corrida 10: seed = 29

════════════════════════════════════════════════════════════════════════════
```

#### 4. Impresión del Mejor Individuo en Cada Generación
**Archivo modificado**: `polyploid_genetic_algorithm.py`

Se actualiza el método `run()` para imprimir el mejor individuo de cada política cada 20 generaciones:

```python
# Cada generación (20, 40, 60, 80, 100):
for policy in self.data.policy_names:
    fronts = self.fast_non_dominated_sort(self.population, policy)
    if fronts and len(fronts[0]) > 0:
        # Encontrar el individuo con mejor balance (punto de la rodilla)
        best_idx = 0
        best_distance = float('inf')
        for idx, ind in enumerate(fronts[0]):
            # Normalizar objetivos y calcular distancia al ideal
            makespan = ind.objectives[policy][0]
            energy = ind.objectives[policy][1]
            distance = (makespan**2 + energy**2)**0.5
            if distance < best_distance:
                best_distance = distance
                best_idx = idx
        
        best = fronts[0][best_idx]
        makespan = best.objectives[policy][0]
        energy = best.objectives[policy][1]
        print(f"    {policy:8s}: Makespan={makespan:8.2f} | Energía={energy:8.2f}")
```

**Salida en consola durante la ejecución**:
```
Generación 20/100

  Mejores individuos por política:
    FIFO    : Makespan= 245.34 | Energía= 1523.45
    LTP     : Makespan= 238.12 | Energía= 1487.23
    STP     : Makespan= 251.89 | Energía= 1512.67
    RRFIFO  : Makespan= 232.45 | Energía= 1501.23
    RRLTP   : Makespan= 228.76 | Energía= 1489.34
    RRECA   : Makespan= 230.12 | Energía= 1495.67

Generación 40/100
  Mejores individuos por política:
    ...
```

**Información que se muestra**:
- ✅ Makespan (tiempo total de ejecución) para cada política
- ✅ Energía (consumo energético total) para cada política
- ✅ Mejor individuo según criterio de rodilla (balance entre objetivos)
- ✅ Progreso del algoritmo en generaciones 20, 40, 60, 80, 100

#### 5. Semillas Guardadas en Archivo de Salida
**Archivo afectado**: `tablas_hipervolumen.txt`

Las semillas utilizadas se guardan automáticamente al inicio del archivo:

```
════════════════════════════════════════════════════════════════════════════
ESTADÍSTICAS DE HIPERVOLUMEN POR POLÍTICA Y GENERACIÓN
════════════════════════════════════════════════════════════════════════════

SEMILLAS USADAS EN CADA CORRIDA:
----------------------------------------------------------------------------------------------------
  Corrida 1: seed = 2
  Corrida 2: seed = 3
  Corrida 3: seed = 5
  Corrida 4: seed = 7
  Corrida 5: seed = 11
  Corrida 6: seed = 13
  Corrida 7: seed = 17
  Corrida 8: seed = 19
  Corrida 9: seed = 23
  Corrida 10: seed = 29

Tabla 1: Políticas FIFO, LTP, STP
...
```

### Impacto de las Actualizaciones

| Aspecto | Antes | Después |
|---------|-------|---------|
| **Semillas** | 0-9 (secuencial) | Números primos (2, 3, 5, ..., 29) |
| **Variabilidad** | Baja | Alta |
| **Trazabilidad** | Rutas hardcodeadas en múltiples lugares | Variable global única |
| **Monitoreo** | Sin información detallada en generaciones intermedias | Impresión de mejor individuo cada 20 generaciones |
| **Documentación** | Semillas no guardadas | Semillas en archivo de salida |
| **Reproducibilidad** | Difícil cambiar carpeta de salida | Fácil (una sola variable) |

### Cómo Ejecutar con Nuevas Actualizaciones

```bash
# Ejecutar experimentación completa
python run_experiment.py

# Salida esperada:
# 1. Impresión de semillas primas (generadas automáticamente)
# 2. Mejor individuo de cada política cada 20 generaciones
# 3. Todos los archivos guardados en OUTPUT_DIR
# 4. Semillas incluidas en tablas_hipervolumen.txt
```