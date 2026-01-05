# EXPLICACIÓN LÍNEA POR LÍNEA DEL CÓDIGO

## 📌 PARTE 1: CLASE JobShopData

### Importaciones y Definición de Clase

```python
import numpy as np  # Para operaciones con matrices y arrays numéricos
import random  # Para generación de números aleatorios
import matplotlib.pyplot as plt  # Para crear gráficas
import matplotlib.patches as mpatches  # Para crear leyendas personalizadas
from copy import deepcopy  # Para copiar objetos profundamente
from typing import List, Tuple, Dict  # Para anotaciones de tipo
import json  # Para manejo de datos JSON
```

**Línea por línea:**
- `numpy`: Biblioteca fundamental para cálculo numérico, usada para matrices de tiempos/energía
- `random`: Genera valores aleatorios para mutaciones, selección, etc.
- `matplotlib`: Crea todas las visualizaciones (Gantt, Pareto)
- `deepcopy`: Evita referencias compartidas al copiar individuos
- `typing`: Mejora legibilidad con tipos explícitos
- `json`: Podría usarse para guardar/cargar configuraciones

### Constructor de JobShopData

```python
def __init__(self):
    self.num_machines = 4  # Define que hay 4 máquinas disponibles (M1, M2, M3, M4)
```

**Explicación:** Establece el número de máquinas del problema. Este valor es fijo según la tabla del PDF.

```python
    self.num_operations = 5  # Hay 5 tipos de operaciones (O1, O2, O3, O4, O5)
```

**Explicación:** Define los 5 tipos diferentes de operaciones que pueden ejecutarse.

```python
    self.jobs = {
        1: [1, 3, 4],        # Trabajo 1 necesita ejecutar O2, O4, O5 (índice base 0)
        2: [0, 2, 4],        # Trabajo 2 necesita ejecutar O1, O3, O5
        3: [0, 1, 2, 3, 4],  # Trabajo 3 necesita todas las operaciones
        4: [3, 4],           # Trabajo 4 solo necesita O4, O5
        5: [1, 3],           # Trabajo 5 necesita O2, O4
        6: [0, 1, 3, 4]      # Trabajo 6 necesita O1, O2, O4, O5
    }
```

**Explicación línea por línea:**
- Diccionario que mapea ID de trabajo → lista de operaciones
- Los índices son base 0: O1=0, O2=1, O3=2, O4=3, O5=4
- Ejemplo: Trabajo 1 debe ejecutar las operaciones en orden: O2 → O4 → O5
- Esta estructura preserva el orden de precedencia de operaciones

```python
    self.num_jobs = len(self.jobs)  # Calcula automáticamente que hay 6 trabajos
```

**Explicación:** Cuenta dinámicamente el número de trabajos, evitando hardcodear valores.

```python
    self.processing_times = np.array([
        [3.5, 6.7, 2.5, 8.2],  # Tiempos de O1 en M1, M2, M3, M4
        [5.5, 4.2, 7.6, 9.0],  # Tiempos de O2 en cada máquina
        [6.1, 7.3, 5.5, 6.7],  # Tiempos de O3
        [4.8, 5.3, 3.8, 4.7],  # Tiempos de O4
        [3.8, 3.4, 4.2, 3.6]   # Tiempos de O5
    ])
```

**Explicación:**
- Matriz 5x4 (operaciones × máquinas)
- `processing_times[i][j]` = tiempo que tarda la operación i en la máquina j
- Ejemplo: O1 en M1 tarda 3.5 unidades, en M2 tarda 6.7
- Estos datos vienen de la Tabla 1 del PDF

```python
    self.energy_consumption = np.array([
        [1.2, 4.7, 3.5, 4.2],  # Consumo energético de O1 en cada máquina
        [7.5, 1.5, 6.6, 3.5],  # O2
        [1.1, 5.3, 8.5, 1.7],  # O3
        [7.8, 3.3, 8.8, 9.7],  # O4
        [1.9, 5.9, 7.5, 3.6]   # O5
    ])
```

**Explicación:**
- Similar a processing_times pero para consumo energético
- `energy_consumption[i][j]` = energía consumida por operación i en máquina j
- Datos de la Tabla 2 del PDF
- Nota: No siempre la máquina más rápida es la más eficiente energéticamente

```python
    self.total_operations = sum(len(ops) for ops in self.jobs.values())
```

**Explicación:**
- Suma el número de operaciones de todos los trabajos
- En este caso: 3 + 3 + 5 + 2 + 2 + 4 = 19 operaciones totales
- Este valor define la longitud de cada cromosoma

```python
    self.policy_names = ['FIFO', 'LTP', 'STP', 'RRFIFO', 'RRLTP', 'RRECA']
```

**Explicación:** Lista con los nombres de las 6 políticas de planificación a evaluar.

```python
    self.policy_orders = self._calculate_policy_orders()
```

**Explicación:** Llama al método privado que calcula el orden de atención para cada política.

## 📌 PARTE 2: Cálculo de Órdenes de Políticas

### Método _calculate_policy_orders

```python
def _calculate_policy_orders(self) -> Dict[str, List[Tuple[int, int]]]:
    orders = {}  # Diccionario vacío para almacenar órdenes
```

**Explicación:** Inicializa diccionario que mapeará: nombre_política → lista_de_operaciones_ordenadas

### Política FIFO

```python
    fifo_order = []  # Lista vacía para el orden FIFO
    for job_id in sorted(self.jobs.keys()):  # Itera trabajos en orden: 1,2,3,4,5,6
        for op_idx in range(len(self.jobs[job_id])):  # Para cada operación del trabajo
            fifo_order.append((job_id, op_idx))  # Añade tupla (trabajo, índice_operación)
    orders['FIFO'] = fifo_order  # Guarda el orden completo
```

**Explicación línea por línea:**
1. Crea lista vacía para almacenar orden
2. Itera trabajos ordenados numéricamente (1, 2, 3, ...)
3. Para cada trabajo, itera sus operaciones en secuencia
4. Añade tupla identificando (ID_trabajo, índice_operación_dentro_del_trabajo)
5. Guarda resultado en diccionario

**Ejemplo del orden resultante:**
```
[(1,0), (1,1), (1,2),  # J1: O2, O4, O5
 (2,0), (2,1), (2,2),  # J2: O1, O3, O5
 (3,0), (3,1), ...]    # J3: todas sus operaciones
```

### Política LTP (Long Time Processing)

```python
    operations_with_time = []  # Lista para almacenar (job, op_idx, tiempo_promedio)
    for job_id, operations in self.jobs.items():  # Itera cada trabajo
        for op_idx, operation in enumerate(operations):  # Itera cada operación del trabajo
            avg_time = np.mean(self.processing_times[operation])  # Calcula tiempo promedio
            operations_with_time.append((job_id, op_idx, avg_time))  # Guarda con su tiempo
```

**Explicación:**
1. Crea lista para almacenar operaciones con su tiempo promedio
2. Itera cada trabajo y sus operaciones
3. `enumerate` proporciona tanto el índice como el valor
4. `np.mean()` calcula el promedio de tiempos en las 4 máquinas
5. Guarda tripleta: (trabajo, índice_op, tiempo_promedio)

**Ejemplo:**
```
operation = 1 (O2)
processing_times[1] = [5.5, 4.2, 7.6, 9.0]
avg_time = (5.5 + 4.2 + 7.6 + 9.0) / 4 = 6.575
```

```python
    operations_with_time.sort(key=lambda x: x[2], reverse=True)  # Ordena por tiempo DESC
```

**Explicación:**
- `sort()` ordena la lista in-place
- `key=lambda x: x[2]` usa el tercer elemento (tiempo) como criterio
- `reverse=True` ordena de mayor a menor (Long Time primero)

```python
    ltp_order = self._apply_precedence_constraints(operations_with_time)
    orders['LTP'] = ltp_order
```

**Explicación:**
- Aplica restricciones de precedencia (Oi debe ir antes que Oi+1)
- Guarda el orden validado en el diccionario

### Método _apply_precedence_constraints

```python
def _apply_precedence_constraints(self, operations_list):
    result = []  # Lista final con orden validado
    completed = {j: 0 for j in self.jobs.keys()}  # Contador de ops completadas por trabajo
```

**Explicación:**
- `result`: Lista donde se construye el orden final válido
- `completed`: Diccionario que rastrea cuántas operaciones de cada trabajo se han programado
- Inicialmente todos en 0 (ninguna operación programada)

```python
    for job_id, op_idx, _ in operations_list:  # Itera operaciones en orden de prioridad
        if op_idx == completed[job_id]:  # Si esta es la siguiente operación esperada
            result.append((job_id, op_idx))  # Programa esta operación
            completed[job_id] += 1  # Incrementa contador de completadas
```

**Explicación:**
- Itera operaciones en su orden de prioridad
- `_` ignora el tercer valor (tiempo/energía)
- Solo programa operación si es la siguiente en secuencia para ese trabajo
- Ejemplo: Si completed[1]=0, solo puede programar J1.O1 (índice 0)

```python
    remaining = []  # Lista de operaciones no programadas aún
    for job_id, op_idx, val in operations_list:
        if (job_id, op_idx) not in result:  # Si no fue programada
            remaining.append((job_id, op_idx, val))  # Añade a pendientes
```

**Explicación:**
- Identifica operaciones que no pudieron programarse en el primer paso
- Esto ocurre cuando sus predecesoras aún no están programadas

```python
    while remaining:  # Mientras haya operaciones pendientes
        added = False  # Flag para detectar si se programó alguna
        for i, (job_id, op_idx, val) in enumerate(remaining):
            if op_idx == completed[job_id]:  # Si ahora es válida
                result.append((job_id, op_idx))  # Programa
                completed[job_id] += 1  # Incrementa contador
                remaining.pop(i)  # Elimina de pendientes
                added = True  # Marca que se programó una
                break  # Sale del for para reiniciar el while
        if not added:  # Si no se pudo programar ninguna
            break  # Sale del while (previene loop infinito)
```

**Explicación:**
- Bucle que intenta programar operaciones restantes
- En cada iteración, busca operaciones que ahora sean válidas
- `enumerate` permite obtener índice para `pop()`
- Si ninguna puede programarse, sale para evitar loop infinito
- Esta lógica garantiza que Oi siempre va antes que Oi+1

```python
    return result  # Retorna lista ordenada validada
```

## 📌 PARTE 3: Clase Individual

### Constructor

```python
def __init__(self, data: JobShopData, chromosomes: Dict[str, np.ndarray] = None):
    self.data = data  # Referencia a los datos del problema
```

**Explicación:** Almacena referencia a datos compartidos (tiempos, energías, etc.)

```python
    if chromosomes is None:  # Si no se proporcionan cromosomas
        self.chromosomes = self._generate_random_chromosomes()  # Genera aleatorios
    else:
        self.chromosomes = chromosomes  # Usa los proporcionados
```

**Explicación:**
- Permite crear individuos de dos formas:
  1. Sin parámetros → genera aleatorio (población inicial)
  2. Con cromosomas → usa existentes (cruza, mutación)

```python
    self.objectives = {}  # Diccionario {policy: (makespan, energy)}
    self.rank = float('inf')  # Nivel de no-dominancia (infinito inicialmente)
    self.crowding_distance = {}  # Diccionario {policy: distancia}
```

**Explicación:**
- `objectives`: Almacena los dos valores objetivo por cada política
- `rank`: Menor es mejor (0 = frente de Pareto, 1 = segundo frente, ...)
- `crowding_distance`: Mide diversidad (mayor = más diverso)

```python
    self._evaluate()  # Calcula objetivos inmediatamente
```

**Explicación:** Evalúa al individuo apenas se crea.

### Generación de Cromosomas Aleatorios

```python
def _generate_random_chromosomes(self) -> Dict[str, np.ndarray]:
    chromosomes = {}  # Diccionario vacío
```

**Explicación:** Inicializa contenedor para los 6 cromosomas.

```python
    for policy in self.data.policy_names:  # Para cada política
        chromosome = np.random.randint(1, self.data.num_machines + 1, 
                                      self.data.total_operations)
        chromosomes[policy] = chromosome
```

**Explicación línea por línea:**
- Itera las 6 políticas
- `np.random.randint(1, 5, 19)`: Genera 19 enteros aleatorios entre 1 y 4
  - 1 a self.num_machines+1 → máquinas 1,2,3,4
  - self.total_operations → 19 genes (uno por operación)
- Cada gen representa la máquina asignada a esa operación
- Guarda cromosoma en diccionario con clave = nombre de política

**Ejemplo de cromosoma generado:**
```
'FIFO': [2, 1, 4, 1, 3, 2, 4, ...]  # 19 números entre 1-4
```

### Evaluación de Objetivos

```python
def _evaluate(self):
    for policy in self.data.policy_names:  # Para cada política
        makespan, energy = self._calculate_objectives(policy)  # Calcula objetivos
        self.objectives[policy] = (makespan, energy)  # Guarda resultados
```

**Explicación:** Calcula y almacena makespan y energía para cada una de las 6 políticas.

### Cálculo de Objetivos (Núcleo del Algoritmo de Programación)

```python
def _calculate_objectives(self, policy: str) -> Tuple[float, float]:
    chromosome = self.chromosomes[policy]  # Obtiene cromosoma de la política
    operation_order = self.data.policy_orders[policy]  # Obtiene orden de operaciones
```

**Explicación:**
- `chromosome`: Array con asignaciones de máquinas
- `operation_order`: Orden en que se procesarán las operaciones

```python
    machine_end_times = np.zeros(self.data.num_machines)  # [0, 0, 0, 0]
```

**Explicación:** Array que rastrea cuándo termina la última operación en cada máquina.

```python
    job_end_times = {j: 0 for j in self.data.jobs.keys()}  # {1:0, 2:0, ..., 6:0}
```

**Explicación:** Diccionario que rastrea cuándo termina la última operación de cada trabajo.

```python
    machine_energy = np.zeros(self.data.num_machines)  # [0, 0, 0, 0]
```

**Explicación:** Array que acumula consumo energético de cada máquina.

```python
    for idx, (job_id, op_idx_in_job) in enumerate(operation_order):
```

**Explicación:**
- Itera cada operación en el orden definido por la política
- `idx`: Posición en la secuencia (0 a 18)
- `job_id`: ID del trabajo (1 a 6)
- `op_idx_in_job`: Índice de operación dentro del trabajo (0, 1, 2, ...)

```python
        operation = self.data.jobs[job_id][op_idx_in_job]  # Obtiene ID real de operación
```

**Explicación:**
- Ejemplo: Si job_id=1 y op_idx_in_job=0, y jobs[1]=[1,3,4]
- Entonces operation=1 (que corresponde a O2 en base 0)

```python
        machine = chromosome[idx] - 1  # Máquina asignada (convierte a base 0)
```

**Explicación:**
- Lee el gen en posición `idx` del cromosoma
- Resta 1 para convertir de base 1 (1-4) a base 0 (0-3)
- Ejemplo: Si chromosome[idx]=3, entonces machine=2 (M3 en base 0)

```python
        proc_time = self.data.processing_times[operation][machine]
        energy = self.data.energy_consumption[operation][machine]
```

**Explicación:**
- Consulta matrices de datos para obtener:
  - `proc_time`: Cuánto tarda esa operación en esa máquina
  - `energy`: Cuánta energía consume

```python
        start_time = max(machine_end_times[machine], job_end_times[job_id])
```

**Explicación CRÍTICA:**
- La operación puede empezar cuando:
  1. La máquina está libre (`machine_end_times[machine]`)
  2. La operación anterior del trabajo terminó (`job_end_times[job_id]`)
- Toma el máximo de ambos (debe cumplir ambas condiciones)

**Ejemplo:**
```
Si máquina M1 termina a tiempo 10
Y última operación de J2 terminó a tiempo 15
Entonces esta operación empieza a tiempo 15 (el mayor)
```

```python
        end_time = start_time + proc_time  # Calcula cuándo termina
```

**Explicación:** Suma tiempo de inicio + duración de procesamiento.

```python
        machine_end_times[machine] = end_time  # Actualiza tiempo de máquina
        job_end_times[job_id] = end_time  # Actualiza tiempo de trabajo
```

**Explicación:**
- Actualiza ambos rastreadores con el nuevo tiempo de finalización
- Esto afectará operaciones futuras en esa máquina o trabajo

```python
        machine_energy[machine] += energy  # Acumula consumo energético
```

**Explicación:** Suma el consumo de esta operación al total de la máquina.

```python
    makespan = np.max(machine_end_times)  # Máximo tiempo entre todas las máquinas
    total_energy = np.sum(machine_energy)  # Suma de consumos de todas las máquinas
    
    return makespan, total_energy
```

**Explicación:**
- `makespan`: Es el tiempo de la máquina que termina más tarde (cuello de botella)
- `total_energy`: Suma simple de consumos de todas las máquinas
- Ambos se minimizan en el algoritmo

## 📌 PARTE 4: Operadores Genéticos

### Cruza Uniforme Poliploide

```python
@staticmethod  # Método estático (no necesita instancia de clase)
def uniform_crossover_polyploid(parent1: Individual, parent2: Individual, 
                                data: JobShopData):
    child1_chromosomes = {}  # Cromosomas del hijo 1
    child2_chromosomes = {}  # Cromosomas del hijo 2
```

**Explicación:**
- `@staticmethod`: No requiere `self`, es una función utilitaria
- Crea diccionarios vacíos para los cromosomas de ambos hijos

```python
    for policy in data.policy_names:  # Para cada política (6 veces)
        mask = np.random.rand(data.total_operations) < 0.5  # Máscara booleana aleatoria
```

**Explicación:**
- `np.random.rand(19)`: Genera 19 números aleatorios entre 0 y 1
- `< 0.5`: Convierte a True/False (aproximadamente 50% True)
- Ejemplo: `[True, False, True, False, ...]`

```python
        child1_chrom = np.where(mask, parent1.chromosomes[policy], 
                               parent2.chromosomes[policy])
```

**Explicación:**
- `np.where(condición, si_verdadero, si_falso)`
- Donde mask es True, toma gen de parent1
- Donde mask es False, toma gen de parent2
- Crea un cromosoma "mosaico" de ambos padres

**Ejemplo visual:**
```
mask:     [T, F, T, F, T]
parent1:  [1, 2, 3, 4, 1]
parent2:  [4, 3, 2, 1, 2]
child1:   [1, 3, 3, 1, 1]  # Toma P1 donde T, P2 donde F
          ↑  ↑  ↑  ↑  ↑
          P1 P2 P1 P2 P1
```

```python
        child2_chrom = np.where(mask, parent2.chromosomes[policy], 
                               parent1.chromosomes[policy])
```

**Explicación:**
- Hijo 2 es el complemento: toma de parent2 donde hijo1 tomó de parent1
- Garantiza que ambos hijos sean diferentes

```python
        child1_chromosomes[policy] = child1_chrom
        child2_chromosomes[policy] = child2_chrom
```

**Explicación:** Almacena cromosomas en sus respectivos diccionarios.

```python
    child1 = Individual(data, child1_chromosomes)  # Crea objeto Individual
    child2 = Individual(data, child2_chromosomes)
    
    return child1, child2  # Retorna ambos hijos
```

**Explicación:**
- Crea nuevos individuos con los cromosomas generados
- El constructor automáticamente evaluará los objetivos
- Retorna tupla con ambos hijos

### Mutación Inter-Cromosoma

```python
@staticmethod
def inter_chromosome_mutation(individual: Individual, data: JobShopData):
    num_swaps = random.choice([2, 3])  # Elige aleatoriamente 2 o 3
```

**Explicación:**
- Decide cuántos cromosomas intercambiar
- `random.choice([2, 3])`: 50% probabilidad de cada opción

```python
    policies_to_swap = random.sample(data.policy_names, num_swaps)
```

**Explicación:**
- Selecciona aleatoriamente 2 o 3 políticas de las 6 disponibles
- `random.sample`: Muestra sin reemplazo (sin repeticiones)
- Ejemplo: ['FIFO', 'STP', 'RRECA']

```python
    if num_swaps == 2:  # Intercambio simple
        p1, p2 = policies_to_swap  # Desempaqueta las dos políticas
        individual.chromosomes[p1], individual.chromosomes[p2] = \
            individual.chromosomes[p2].copy(), individual.chromosomes[p1].copy()
```

**Explicación:**
- Intercambio simultáneo de dos cromosomas
- `.copy()` es CRUCIAL para evitar aliases (referencias compartidas)
- Sintaxis Python para swap: `a, b = b, a`

**Ejemplo:**
```
Antes:
  FIFO: [1,2,3,4]
  STP:  [4,3,2,1]

Después:
  FIFO: [4,3,2,1]
  STP:  [1,2,3,4]
```

```python
    else:  # num_swaps == 3, intercambio circular
        p1, p2, p3 = policies_to_swap
        temp = individual.chromosomes[p1].copy()  # Guarda p1 temporalmente
        individual.chromosomes[p1] = individual.chromosomes[p2].copy()  # p1 ← p2
        individual.chromosomes[p2] = individual.chromosomes[p3].copy()  # p2 ← p3
        individual.chromosomes[p3] = temp  # p3 ← p1 (del temp)
```

**Explicación:**
- Rotación circular: p1 → p2, p2 → p3, p3 → p1
- Requiere variable temporal para no perder p1 original

**Ejemplo visual:**
```
Antes:      Después:
p1: [A]     p1: [B]
p2: [B]  →  p2: [C]
p3: [C]     p3: [A]
```

```python
    individual._evaluate()  # Re-calcula objetivos con nuevos cromosomas
```

**Explicación:**
- CRÍTICO: Después de cualquier mutación, hay que recalcular objetivos
- Los nuevos cromosomas probablemente tengan diferentes makespan/energía

### Mutación por Intercambio Recíproco

```python
@staticmethod
def reciprocal_exchange_mutation(individual: Individual, data: JobShopData, 
                                 num_swaps: int = 2):
```

**Explicación:** Parámetro `num_swaps` define cuántos pares intercambiar (default=2).

```python
    for policy in data.policy_names:  # Para cada política
        chromosome = individual.chromosomes[policy]  # Referencia al cromosoma
```

**Explicación:**
- Itera cada uno de los 6 cromosomas
- `chromosome` es una referencia (no copia), modificar afecta al original

```python
        for _ in range(num_swaps):  # Repetir num_swaps veces
            pos1, pos2 = random.sample(range(data.total_operations), 2)
```

**Explicación:**
- `range(19)`: Números de 0 a 18 (posiciones válidas en cromosoma)
- `random.sample(..., 2)`: Selecciona 2 posiciones diferentes
- Ejemplo: pos1=3, pos2=15

```python
            chromosome[pos1], chromosome[pos2] = chromosome[pos2], chromosome[pos1]
```

**Explicación:**
- Intercambio de valores en esas dos posiciones
- Swap simultáneo de Python

**Ejemplo:**
```
Antes:     [1, 2, 3, 4, 5]
                ↑     ↑
           pos1=2  pos2=4

Después:   [1, 2, 5, 4, 3]
```

```python
    individual._evaluate()  # Re-evalúa con cromosomas modificados
```

### Mutación por Desplazamiento

```python
@staticmethod
def displacement_mutation(individual: Individual, data: JobShopData, 
                         segment_length: int = 3):
```

**Explicación:** `segment_length` define tamaño del segmento a mover (default=3).

```python
    for policy in data.policy_names:
        chromosome = individual.chromosomes[policy]
```

**Explicación:** Itera cada cromosoma para aplicar mutación.

```python
        start_pos = random.randint(0, data.total_operations - segment_length)
```

**Explicación:**
- Selecciona posición inicial del segmento
- Límite superior asegura que el segmento quepa
- Ejemplo: Si total=19 y segment=3, start puede ser 0-16

```python
        segment = chromosome[start_pos:start_pos + segment_length].copy()
```

**Explicación:**
- Extrae segmento de genes
- `copy()` crea nueva copia independiente
- Ejemplo: Si start=5 y length=3, extrae posiciones 5,6,7

```python
        possible_positions = list(range(0, start_pos)) + \
                           list(range(start_pos + segment_length, 
                                    data.total_operations - segment_length + 1))
```

**Explicación detallada:**
- Calcula posiciones válidas para insertar segmento
- `range(0, start_pos)`: Posiciones antes del segmento actual
- `range(start_pos + segment_length, ...)`: Posiciones después
- Excluye posición actual y posiciones donde no cabría

**Ejemplo:**
```
Cromosoma: [0,1,2,3,4,5,6,7,8]  (9 elementos)
Segmento:   [   X,X,X   ]      (posiciones 3-5, length=3)

Posiciones válidas:
  - Antes: [0,1,2]              (range(0,3))
  - Después: [6]                 (range(6,7))
  - NO válidas: 3,4,5 (actual), 7,8 (segmento no cabría)
```

```python
        if possible_positions:  # Si hay posiciones válidas
            new_pos = random.choice(possible_positions)  # Selecciona una
```

**Explicación:**
- Verifica que haya al menos una posición válida
- Selecciona aleatoriamente una de ellas

```python
            remaining = np.delete(chromosome, range(start_pos, start_pos + segment_length))
```

**Explicación:**
- `np.delete`: Elimina elementos en rango especificado
- `remaining`: Cromosoma sin el segmento
- Ejemplo: `[1,2,3,4,5]` → elimina pos 1-2 → `[1,4,5]`

```python
            chromosome = np.insert(remaining, new_pos, segment)
```

**Explicación:**
- `np.insert(array, posición, valores)`: Inserta valores en posición
- Inserta el segmento completo en la nueva posición

**Ejemplo completo:**
```
Original:   [A,B,C,D,E,F,G]
Segmento:      [C,D,E]  (pos 2-4)
Remaining:  [A,B,F,G]
Nueva pos:  1
Resultado:  [A,C,D,E,B,F,G]
```

```python
            individual.chromosomes[policy] = chromosome  # Actualiza cromosoma
```

**Explicación:** Reemplaza el cromosoma modificado en el individuo.

```python
    individual._evaluate()  # Re-calcula objetivos
```

---

## 📌 PARTE 5: Algoritmo NSGA-II (Núcleo Principal)

### Ordenamiento No-Dominado Rápido

```python
def fast_non_dominated_sort(self, population: List[Individual], policy: str):
    S = [[] for _ in range(len(population))]  # S[i] = índices dominados por i
    n = [0] * len(population)  # n[i] = cuántos dominan a i
    rank = [0] * len(population)  # rank[i] = nivel de i
    fronts = [[]]  # fronts[0] = primer frente
```

**Explicación:**
- `S[i]`: Lista de individuos que el individuo i domina
- `n[i]`: Contador de cuántos individuos dominan a i
- `rank[i]`: Nivel de no-dominancia (0=mejor)
- `fronts`: Lista de listas, cada sublista es un frente

```python
    for i in range(len(population)):  # Para cada individuo i
        for j in range(len(population)):  # Comparar con cada j
            if i != j:  # No comparar consigo mismo
```

**Explicación:** Doble bucle para comparar todos los pares.

```python
                if population[i].dominates(population[j], policy):
                    S[i].append(j)  # i domina a j, añadir j a lista de i
```

**Explicación:**
- Llama al método `dominates` del individuo
- Si i es mejor que j en ambos objetivos, i domina a j

```python
                elif population[j].dominates(population[i], policy):
                    n[i] += 1  # j domina a i, incrementar contador
```

**Explicación:**
- Si j domina a i, incrementa cuántos dominan a i

```python
        if n[i] == 0:  # Si nadie domina a i
            rank[i] = 0  # Asigna rank 0
            fronts[0].append(i)  # Añade al primer frente
```

**Explicación:**
- Si nadie domina a i, i está en el frente de Pareto (rank 0)

```python
    i = 0  # Índice de frente actual
    while i < len(fronts) and fronts[i]:  # Mientras haya frentes con elementos
        next_front = []  # Inicializa siguiente frente
```

**Explicación:**
- Itera frentes existentes
- `fronts[i]` verifica que el frente no esté vacío

```python
        for p_idx in fronts[i]:  # Para cada individuo p en frente actual
            for q_idx in S[p_idx]:  # Para cada q dominado por p
                n[q_idx] -= 1  # Reduce contador de q
```

**Explicación:**
- Cuando p se asigna a un frente, "libera" a los que domina
- Decrementa sus contadores

```python
                if n[q_idx] == 0:  # Si ahora nadie domina a q
                    rank[q_idx] = i + 1  # Asigna siguiente rank
                    next_front.append(q_idx)  # Añade a siguiente frente
```

**Explicación:**
- Si contador llega a 0, q va al siguiente frente
- rank es i+1 (un nivel peor que frente actual)

```python
        i += 1  # Avanza al siguiente frente
        if next_front:  # Si hay elementos en siguiente frente
            fronts.append(next_front)  # Añade frente a la lista
```

**Explicación:**
- Solo añade frente si tiene elementos
- Avanza al siguiente nivel

```python
    result_fronts = []  # Convertir índices a individuos
    for front in fronts:
        if front:
            result_fronts.append([population[idx] for idx in front])
    
    return result_fronts
```

**Explicación:**
- Convierte listas de índices a listas de objetos Individual
- Retorna lista de listas (frentes)

### Cálculo de Distancia de Crowding

```python
def calculate_crowding_distance(self, front: List[Individual], policy: str):
    if len(front) == 0:  # Si frente vacío
        return  # No hace nada
```

**Explicación:** Validación para evitar errores con frentes vacíos.

```python
    for ind in front:  # Inicializa todas las distancias a 0
        ind.crowding_distance[policy] = 0
```

**Explicación:** Resetea distancias antes de calcular.

```python
    if len(front) <= 2:  # Si hay 2 o menos individuos
        for ind in front:
            ind.crowding_distance[policy] = float('inf')  # Asigna infinito
        return
```

**Explicación:**
- Con ≤2 individuos, todos son extremos
- Distancia infinita asegura que se preserven

```python
    for obj_idx in range(2):  # Para cada objetivo (makespan y energía)
        front.sort(key=lambda x: x.objectives[policy][obj_idx])
```

**Explicación:**
- Ordena frente por objetivo actual
- obj_idx=0: ordena por makespan
- obj_idx=1: ordena por energía

```python
        front[0].crowding_distance[policy] = float('inf')  # Extremo inferior
        front[-1].crowding_distance[policy] = float('inf')  # Extremo superior
```

**Explicación:**
- Individuos con mejor y peor valor en objetivo tienen distancia infinita
- Esto preserva la diversidad en los extremos

```python
        obj_range = (front[-1].objectives[policy][obj_idx] - 
                    front[0].objectives[policy][obj_idx])
```

**Explicación:**
- Calcula rango del objetivo
- Diferencia entre mejor y peor valor

```python
        if obj_range == 0:  # Si todos tienen mismo valor
            continue  # Salta a siguiente objetivo
```

**Explicación:** Evita división por cero si todos son iguales.

```python
        for i in range(1, len(front) - 1):  # Para individuos intermedios
            distance = (front[i + 1].objectives[policy][obj_idx] - 
                       front[i - 1].objectives[policy][obj_idx]) / obj_range
            front[i].crowding_distance[policy] += distance
```

**Explicación:**
- Calcula distancia normalizada entre vecinos
- `(siguiente - anterior) / rango_total`
- Suma distancias de ambos objetivos
- Mayor distancia = más aislado = más diverso

**Ejemplo visual:**
```
Makespan: |---*---*---------*-----|
           A   B             C
           
Distance(B) = (C - A) / (max - min)
            = (distancia entre vecinos) / (rango total)
```

Esta explicación cubre aproximadamente el 50% del código. ¿Te gustaría que continúe con:
1. La selección por torneo con intercambio cromosómico
2. El método `run()` principal
3. El cálculo de hipervolumen
4. Las funciones de visualización

Por favor indícame qué sección te gustaría ver a continuación y seguiré con el mismo nivel de detalle.
