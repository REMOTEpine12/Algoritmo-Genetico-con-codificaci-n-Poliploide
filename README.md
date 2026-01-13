# ALGORITMO GENÉTICO POLIPLOIDE PARA PLANIFICACIÓN DE TAREAS

## Descripción General

Este proyecto implementa un **Algoritmo Genético con codificación Poliploide** usando **NSGA-II** para resolver el problema de planificación de tareas (Job Shop Scheduling Problem) optimizando dos objetivos:
1. **Makespan** (tiempo total de ejecución)
2. **Consumo energético total**

## Estructura del Código

### 1. Clase `JobShopData`
**Propósito**: Almacenar y gestionar todos los datos del problema.

**Atributos principales**:
- `processing_times`: Matriz de tiempos de procesamiento (operación x máquina)
- `energy_consumption`: Matriz de consumo energético (operación x máquina)
- `jobs`: Diccionario que define qué operaciones tiene cada trabajo
- `policy_orders`: Orden de atención de operaciones según cada política

**Métodos clave**:
- `_calculate_policy_orders()`: Calcula el orden en que se atenderán las operaciones según cada una de las 6 políticas

### 2. Clase `Individual`
**Propósito**: Representa una solución (individuo) en el algoritmo genético.

**Codificación Poliploide**:
- Cada individuo tiene **6 cromosomas** (uno por política)
- Cada cromosoma es un array de enteros
- Cada gen representa la **máquina asignada** a una operación
- Ejemplo: `[1, 4, 2, 3, ...]` significa que la operación 1 va a máquina 1, operación 2 a máquina 4, etc.

**Métodos clave**:
- `_generate_random_chromosomes()`: Genera cromosomas aleatorios iniciales
- `_calculate_objectives()`: Calcula makespan y energía para una política
- `dominates()`: Determina si un individuo domina a otro (para Pareto)

### 3. Clase `GeneticOperators`
**Propósito**: Implementa todos los operadores genéticos.

#### a) Cruza Uniforme Poliploide
```python
uniform_crossover_polyploid(parent1, parent2)
```
- Crea una máscara aleatoria para cada cromosoma
- Cada hijo toma genes alternadamente de cada padre
- Se aplica a TODOS los cromosomas simultáneamente

#### b) Mutación Inter-Cromosoma
```python
inter_chromosome_mutation(individual)
```
- Intercambia 2 o 3 cromosomas completos entre políticas
- Ejemplo: FIFO ↔ LTP o FIFO → LTP → STP → FIFO
- Cambia el valor de las funciones objetivo

#### c) Mutación por Intercambio Recíproco
```python
reciprocal_exchange_mutation(individual, num_swaps=2)
```
- Selecciona pares aleatorios de genes en cada cromosoma
- Intercambia sus posiciones
- Se aplica a cada cromosoma independientemente

#### d) Mutación por Desplazamiento
```python
displacement_mutation(individual, segment_length=3)
```
- Selecciona un segmento de genes
- Lo mueve a otra posición (rotación circular)
- Mantiene el orden relativo del segmento

### 4. Clase `PolyploidNSGAII`
**Propósito**: Implementa el algoritmo NSGA-II completo.

#### Algoritmo Principal
```
1. Inicializar población P(0)
2. Para cada generación t:
   a. Crear descendientes Q(t) mediante:
      - Selección por torneo
      - Cruza uniforme
      - Mutación
   b. Combinar R(t) = P(t) ∪ Q(t)
   c. Ordenamiento no-dominado de R(t)
   d. Calcular distancia de crowding
   e. Seleccionar mejores N individuos para P(t+1)
3. Retornar población final
```

**Métodos clave**:

#### `fast_non_dominated_sort(population, policy)`
Implementa el algoritmo de ordenamiento rápido no-dominado:
1. Para cada par de individuos, determina dominancia
2. Agrupa individuos en frentes (F1, F2, F3...)
3. F1 contiene soluciones no-dominadas
4. F2 contiene soluciones dominadas solo por F1, etc.

#### `calculate_crowding_distance(front, policy)`
Calcula la distancia de crowding (diversidad):
1. Ordena el frente por cada objetivo
2. Asigna distancia infinita a los extremos
3. Para individuos intermedios:
   - distance = (valor_siguiente - valor_anterior) / rango_objetivo
4. Suma distancias de ambos objetivos

#### `tournament_selection_with_chromosome_exchange(population)`
Selección especial con intercambio cromosómico:
1. Selecciona 2 individuos aleatorios
2. Para cada política:
   - Compara basándose en rank y crowding distance
   - Selecciona el mejor cromosoma de ese política
3. Crea un "super-individuo" con los mejores cromosomas

#### `calculate_hypervolume(front, policy)`
Calcula el hipervolumen del frente de Pareto:
- Mide la región del espacio objetivo dominada por el frente
- Punto de referencia: 10% peor que el peor valor encontrado
- Algoritmo de barrido para cálculo eficiente

## Políticas de Atención Implementadas

### 1. FIFO (First In, First Out)
Las operaciones se atienden en el orden de llegada (orden de trabajo):
```
J1 → J2 → J3 → J4 → J5 → J6
```

### 2. LTP (Long Time Processing)
Prioriza operaciones con mayor tiempo promedio:
1. Calcula tiempo promedio de cada operación
2. Ordena descendentemente
3. Aplica restricciones de precedencia

### 3. STP (Short Time Processing)
Prioriza operaciones con menor tiempo promedio:
1. Calcula tiempo promedio de cada operación
2. Ordena ascendentemente
3. Aplica restricciones de precedencia

### 4. RRFIFO (Round Robin + FIFO)
Alterna entre trabajos de forma circular:
```
J1.O1 → J2.O1 → J3.O1 → J4.O1 → J5.O1 → J6.O1 →
J1.O2 → J2.O2 → J3.O2 → ...
```

### 5. RRLTP (Round Robin + Long Time Processing)
Round Robin pero ordenando trabajos por tiempo promedio descendente.

### 6. RRECA (Round Robin + Energy Consumption Average)
Round Robin pero ordenando trabajos por consumo energético ascendente.

## Funciones Objetivo

### F1: Makespan (Minimizar)
```
makespan = max{tiempo_final_máquina_i | i ∈ [1, m]}
```
- Tiempo total hasta que se completan todos los trabajos
- Equivale al tiempo de la máquina que termina más tarde

### F2: Consumo Energético Total (Minimizar)
```
energía_total = Σ(energía_consumida_por_máquina_i)
```
- Suma del consumo de todas las máquinas
- Incluye solo el consumo durante procesamiento

## Características Especiales

### Restricciones Respetadas:
1. Las operaciones de un trabajo deben ejecutarse en orden
2. Una operación no puede reasignarse una vez programada
3. Una máquina solo puede procesar una operación a la vez
4. No hay tiempos de setup entre operaciones

### Optimización Multi-Objetivo:
- Usa concepto de dominancia de Pareto
- Mantiene diversidad con crowding distance
- Genera frente de Pareto con múltiples soluciones de compromiso

## Métricas de Evaluación

### Hipervolumen
- Mide la calidad del frente de Pareto
- Mayor hipervolumen = mejor convergencia y diversidad
- Se calcula para cada política en generaciones 20, 40, 60, 80, 100

### Solución de la Rodilla
- Punto de mejor compromiso entre objetivos
- Se encuentra como el punto más cercano al ideal (0, 0) normalizado
- Representa un equilibrio óptimo entre makespan y energía

## Uso del Código

### Prueba Rápida
```bash
python test_algorithm.py
```
Ejecuta una prueba con parámetros reducidos (10 individuos, 20 generaciones).

### Experimentación Completa
```bash
python run_experiment.py
```
Ejecuta 10 corridas completas con 100 generaciones cada una.

### Uso Programático
```python
from polyploid_genetic_algorithm import *

# Cargar datos
data = JobShopData()

# Crear algoritmo
algorithm = PolyploidNSGAII(
    data=data,
    population_size=20,
    generations=100,
    crossover_rate=0.8,
    seed=42
)

# Ejecutar
final_population = algorithm.run()

# Obtener resultados
for policy in data.policy_names:
    pareto_front = algorithm.get_pareto_front(policy)
    knee_solution = algorithm.find_knee_solution(policy)
    
    # Visualizar
    plot_pareto_front(algorithm, policy, f"pareto_{policy}.png")
    create_gantt_chart(knee_solution, policy, data, f"gantt_{policy}.png")
```

## Archivos Generados

### Imágenes
- `pareto_{POLICY}.png`: Frente de Pareto para cada política
- `gantt_{POLICY}_knee.png`: Diagrama de Gantt de la solución de rodilla
- `gantt_{POLICY}_min_makespan.png`: Gantt con menor makespan
- `gantt_{POLICY}_min_energy.png`: Gantt con menor energía

### Reportes
- `report_{POLICY}.txt`: Tabla con asignaciones y objetivos

## Parámetros Recomendados

```python
population_size = 20       # Tamaño de población
generations = 100          # Número de generaciones
crossover_rate = 0.8       # Probabilidad de cruza

mutation_rates = {
    'inter_chromosome': 0.3,      # Mutación inter-cromosoma
    'reciprocal_exchange': 0.2,   # Intercambio recíproco
    'displacement': 0.1           # Desplazamiento
}
```

## Ventajas de la Codificación Poliploide

1. **Exploración paralela**: Cada cromosoma explora con diferente política
2. **Diversidad natural**: Múltiples representaciones en un solo individuo
3. **Información redundante**: Mayor robustez ante mutaciones
4. **Especialización**: Cada cromosoma puede especializarse en su política

## Referencias

- Deb, K., et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II
- Algoritmos genéticos poliploides para optimización multi-objetivo
- Job Shop Scheduling Problem: formulaciones y métodos de solución

## Autor
- [@Remotepine99](https://github.com/REMOTEpine12) - Israel Díaz
- [@AtzMax](https://github.com/AtzMax) - Atzin Ignacio
