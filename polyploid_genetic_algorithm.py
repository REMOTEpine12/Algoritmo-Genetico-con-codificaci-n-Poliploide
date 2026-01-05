"""
=================================================================================
ALGORITMO GENÉTICO POLIPLOIDE NSGA-II PARA PLANIFICACIÓN DE TAREAS
=================================================================================
Práctica 2: Algoritmo genético poliploide para planificación de tareas

Autor: Implementación de la práctica
Descripción: Este código implementa un algoritmo genético con codificación
             poliploide usando NSGA-II para optimizar la planificación de
             tareas considerando makespan y consumo energético.
=================================================================================
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from copy import deepcopy
from typing import List, Tuple, Dict
import json

# =============================================================================
# CLASE PARA MANEJAR LOS DATOS DEL PROBLEMA
# =============================================================================

class JobShopData:
    """
    Clase que almacena y gestiona los datos del problema de planificación de tareas.
    Incluye información sobre trabajos, operaciones, máquinas, tiempos y consumo energético.
    """
    
    def __init__(self):
        """
        Inicializa los datos del problema según las tablas del documento.
        """
        # Número de máquinas disponibles
        self.num_machines = 4
        
        # Número de operaciones totales
        self.num_operations = 5
        
        # Definición de trabajos y sus operaciones
        # Cada trabajo es una lista de operaciones (índice base 0)
        self.jobs = {
            1: [1, 3, 4],        # J1: {O2, O4, O5}
            2: [0, 2, 4],        # J2: {O1, O3, O5}
            3: [0, 1, 2, 3, 4],  # J3: {O1, O2, O3, O4, O5}
            4: [3, 4],           # J4: {O4, O5}
            5: [1, 3],           # J5: {O2, O4}
            6: [0, 1, 3, 4]      # J6: {O1, O2, O4, O5}
        }
        
        # Número total de trabajos
        self.num_jobs = len(self.jobs)
        
        # Tabla de tiempos: tiempo[operacion][maquina]
        # Tiempos de procesamiento por operación en cada máquina
        self.processing_times = np.array([
            [3.5, 6.7, 2.5, 8.2],  # O1
            [5.5, 4.2, 7.6, 9.0],  # O2
            [6.1, 7.3, 5.5, 6.7],  # O3
            [4.8, 5.3, 3.8, 4.7],  # O4
            [3.8, 3.4, 4.2, 3.6]   # O5
        ])
        
        # Tabla de consumo energético: energy[operacion][maquina]
        # Consumo energético por operación en cada máquina
        self.energy_consumption = np.array([
            [1.2, 4.7, 3.5, 4.2],  # O1
            [7.5, 1.5, 6.6, 3.5],  # O2
            [1.1, 5.3, 8.5, 1.7],  # O3
            [7.8, 3.3, 8.8, 9.7],  # O4
            [1.9, 5.9, 7.5, 3.6]   # O5
        ])
        
        # Calcular el número total de operaciones a planificar
        self.total_operations = sum(len(ops) for ops in self.jobs.values())
        
        # Nombres de las políticas
        self.policy_names = ['FIFO', 'LTP', 'STP', 'RRFIFO', 'RRLTP', 'RRECA']
        
        # Calcular los órdenes de atención según cada política
        self.policy_orders = self._calculate_policy_orders()
    
    def _calculate_policy_orders(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Calcula el orden de atención de las operaciones según cada política.
        
        Returns:
            Diccionario con el orden de operaciones para cada política.
            Cada operación se representa como (job_id, operation_index_in_job)
        """
        orders = {}
        
        # =================================================================
        # POLÍTICA 1: FIFO (First In First Out)
        # Las operaciones se atienden en el orden en que llegan (orden de trabajo)
        # =================================================================
        fifo_order = []
        for job_id in sorted(self.jobs.keys()):
            for op_idx in range(len(self.jobs[job_id])):
                fifo_order.append((job_id, op_idx))
        orders['FIFO'] = fifo_order
        
        # =================================================================
        # POLÍTICA 2: LTP (Long Time Processing)
        # Prioriza operaciones con mayor tiempo promedio de procesamiento
        # =================================================================
        # Crear lista de (job_id, op_idx_in_job, tiempo_promedio)
        operations_with_time = []
        for job_id, operations in self.jobs.items():
            for op_idx, operation in enumerate(operations):
                avg_time = np.mean(self.processing_times[operation])
                operations_with_time.append((job_id, op_idx, avg_time))
        
        # Ordenar por tiempo promedio descendente (mayor tiempo primero)
        operations_with_time.sort(key=lambda x: x[2], reverse=True)
        
        # Crear orden manteniendo restricciones de precedencia
        ltp_order = self._apply_precedence_constraints(operations_with_time)
        orders['LTP'] = ltp_order
        
        # =================================================================
        # POLÍTICA 3: STP (Short Time Processing)
        # Prioriza operaciones con menor tiempo promedio de procesamiento
        # =================================================================
        # Ordenar por tiempo promedio ascendente (menor tiempo primero)
        operations_with_time.sort(key=lambda x: x[2])
        stp_order = self._apply_precedence_constraints(operations_with_time)
        orders['STP'] = stp_order
        
        # =================================================================
        # POLÍTICA 4: RRFIFO (Round Robin + FIFO)
        # Alterna entre trabajos en orden circular, respetando FIFO
        # =================================================================
        rrfifo_order = []
        job_pointers = {j: 0 for j in self.jobs.keys()}  # Puntero de operación por trabajo
        
        # Mientras haya operaciones pendientes
        while any(job_pointers[j] < len(self.jobs[j]) for j in self.jobs.keys()):
            for job_id in sorted(self.jobs.keys()):
                # Si el trabajo aún tiene operaciones pendientes
                if job_pointers[job_id] < len(self.jobs[job_id]):
                    rrfifo_order.append((job_id, job_pointers[job_id]))
                    job_pointers[job_id] += 1
        orders['RRFIFO'] = rrfifo_order
        
        # =================================================================
        # POLÍTICA 5: RRLTP (Round Robin + Long Time Processing)
        # Round Robin pero priorizando trabajos con operaciones más largas
        # =================================================================
        # Calcular tiempo promedio total por trabajo
        job_avg_times = []
        for job_id, operations in self.jobs.items():
            total_time = sum(np.mean(self.processing_times[op]) for op in operations)
            job_avg_times.append((job_id, total_time / len(operations)))
        
        # Ordenar trabajos por tiempo promedio descendente
        job_avg_times.sort(key=lambda x: x[1], reverse=True)
        job_order = [j[0] for j in job_avg_times]
        
        # Aplicar Round Robin con este orden
        rrltp_order = []
        job_pointers = {j: 0 for j in self.jobs.keys()}
        
        while any(job_pointers[j] < len(self.jobs[j]) for j in self.jobs.keys()):
            for job_id in job_order:
                if job_pointers[job_id] < len(self.jobs[job_id]):
                    rrltp_order.append((job_id, job_pointers[job_id]))
                    job_pointers[job_id] += 1
        orders['RRLTP'] = rrltp_order
        
        # =================================================================
        # POLÍTICA 6: RRECA (Round Robin + Energy Consumption Average)
        # Round Robin priorizando trabajos con menor consumo energético promedio
        # =================================================================
        # Calcular consumo energético promedio por trabajo
        job_avg_energy = []
        for job_id, operations in self.jobs.items():
            total_energy = sum(np.mean(self.energy_consumption[op]) for op in operations)
            job_avg_energy.append((job_id, total_energy / len(operations)))
        
        # Ordenar trabajos por consumo energético promedio ascendente
        job_avg_energy.sort(key=lambda x: x[1])
        job_order = [j[0] for j in job_avg_energy]
        
        # Aplicar Round Robin con este orden
        rreca_order = []
        job_pointers = {j: 0 for j in self.jobs.keys()}
        
        while any(job_pointers[j] < len(self.jobs[j]) for j in self.jobs.keys()):
            for job_id in job_order:
                if job_pointers[job_id] < len(self.jobs[job_id]):
                    rreca_order.append((job_id, job_pointers[job_id]))
                    job_pointers[job_id] += 1
        orders['RRECA'] = rreca_order
        
        return orders
    
    def _apply_precedence_constraints(self, operations_list: List[Tuple]) -> List[Tuple[int, int]]:
        """
        Aplica restricciones de precedencia: Oi,j debe ejecutarse antes que O(i+1),j.
        
        Args:
            operations_list: Lista de tuplas (job_id, op_idx, valor_ordenamiento)
        
        Returns:
            Lista ordenada respetando precedencias
        """
        result = []
        completed = {j: 0 for j in self.jobs.keys()}  # Operaciones completadas por trabajo
        
        # Procesar operaciones en el orden dado, pero respetando precedencias
        for job_id, op_idx, _ in operations_list:
            # Solo añadir si todas las operaciones previas del trabajo están completadas
            if op_idx == completed[job_id]:
                result.append((job_id, op_idx))
                completed[job_id] += 1
        
        # Añadir operaciones restantes que no pudieron agregarse
        remaining = []
        for job_id, op_idx, val in operations_list:
            if (job_id, op_idx) not in result:
                remaining.append((job_id, op_idx, val))
        
        # Procesar operaciones restantes respetando precedencias
        while remaining:
            added = False
            for i, (job_id, op_idx, val) in enumerate(remaining):
                if op_idx == completed[job_id]:
                    result.append((job_id, op_idx))
                    completed[job_id] += 1
                    remaining.pop(i)
                    added = True
                    break
            if not added:
                break
        
        return result


# =============================================================================
# CLASE PARA REPRESENTAR UN INDIVIDUO (SOLUCIÓN)
# =============================================================================

class Individual:
    """
    Representa un individuo en el algoritmo genético poliploide.
    Cada individuo tiene 6 cromosomas (uno por política).
    """
    
    def __init__(self, data: JobShopData, chromosomes: Dict[str, np.ndarray] = None):
        """
        Inicializa un individuo.
        
        Args:
            data: Objeto con los datos del problema
            chromosomes: Diccionario de cromosomas (si None, se genera aleatoriamente)
        """
        self.data = data
        
        # Si no se proporcionan cromosomas, generar aleatoriamente
        if chromosomes is None:
            self.chromosomes = self._generate_random_chromosomes()
        else:
            self.chromosomes = chromosomes
        
        # Inicializar métricas
        self.objectives = {}  # {policy: (makespan, energy)}
        self.rank = float('inf')  # Nivel de no-dominancia
        self.crowding_distance = {}  # {policy: distance}
        
        # Calcular objetivos para cada política
        self._evaluate()
    
    def _generate_random_chromosomes(self) -> Dict[str, np.ndarray]:
        """
        Genera cromosomas aleatorios para cada política.
        Cada gen representa la máquina asignada a una operación.
        
        Returns:
            Diccionario con un cromosoma por política
        """
        chromosomes = {}
        
        # Para cada política, generar un cromosoma aleatorio
        for policy in self.data.policy_names:
            # Cada cromosoma tiene longitud igual al número total de operaciones
            # Cada gen es un número entre 1 y num_machines (máquina asignada)
            chromosome = np.random.randint(1, self.data.num_machines + 1, 
                                          self.data.total_operations)
            chromosomes[policy] = chromosome
        
        return chromosomes
    
    def _evaluate(self):
        """
        Evalúa las funciones objetivo (makespan y energía) para cada cromosoma/política.
        """
        # Evaluar cada política
        for policy in self.data.policy_names:
            makespan, energy = self._calculate_objectives(policy)
            self.objectives[policy] = (makespan, energy)
    
    def _calculate_objectives(self, policy: str) -> Tuple[float, float]:
        """
        Calcula makespan y consumo energético para una política específica.
        
        Args:
            policy: Nombre de la política
        
        Returns:
            Tupla (makespan, consumo_energético_total)
        """
        # Obtener cromosoma y orden de operaciones para esta política
        chromosome = self.chromosomes[policy]
        operation_order = self.data.policy_orders[policy]
        
        # Inicializar tiempos de finalización por máquina
        machine_end_times = np.zeros(self.data.num_machines)
        
        # Inicializar tiempo de finalización de última operación por trabajo
        job_end_times = {j: 0 for j in self.data.jobs.keys()}
        
        # Inicializar consumo energético por máquina
        machine_energy = np.zeros(self.data.num_machines)
        
        # Procesar cada operación en el orden definido por la política
        for idx, (job_id, op_idx_in_job) in enumerate(operation_order):
            # Obtener el índice real de la operación (O1, O2, etc.)
            operation = self.data.jobs[job_id][op_idx_in_job]
            
            # Obtener máquina asignada (índice base 1, convertir a base 0)
            machine = chromosome[idx] - 1
            
            # Obtener tiempo de procesamiento y consumo energético
            proc_time = self.data.processing_times[operation][machine]
            energy = self.data.energy_consumption[operation][machine]
            
            # Calcular tiempo de inicio: máximo entre tiempo libre de máquina
            # y tiempo de finalización de operación anterior del mismo trabajo
            start_time = max(machine_end_times[machine], job_end_times[job_id])
            
            # Calcular tiempo de finalización
            end_time = start_time + proc_time
            
            # Actualizar tiempos
            machine_end_times[machine] = end_time
            job_end_times[job_id] = end_time
            
            # Actualizar consumo energético
            machine_energy[machine] += energy
        
        # Makespan es el tiempo máximo de todas las máquinas
        makespan = np.max(machine_end_times)
        
        # Consumo energético total es la suma de todas las máquinas
        total_energy = np.sum(machine_energy)
        
        return makespan, total_energy
    
    def dominates(self, other: 'Individual', policy: str) -> bool:
        """
        Verifica si este individuo domina a otro en una política específica.
        Un individuo A domina a B si es mejor o igual en todos los objetivos
        y estrictamente mejor en al menos uno.
        
        Args:
            other: Otro individuo
            policy: Política a comparar
        
        Returns:
            True si este individuo domina al otro
        """
        # Obtener objetivos
        self_obj = self.objectives[policy]
        other_obj = other.objectives[policy]
        
        # Verificar dominancia (minimizamos ambos objetivos)
        better_in_one = False
        for i in range(2):
            if self_obj[i] > other_obj[i]:
                return False  # Peor en al menos un objetivo
            if self_obj[i] < other_obj[i]:
                better_in_one = True
        
        return better_in_one


# =============================================================================
# OPERADORES GENÉTICOS
# =============================================================================

class GeneticOperators:
    """
    Clase que implementa los operadores genéticos para el algoritmo poliploide.
    """
    
    @staticmethod
    def uniform_crossover_polyploid(parent1: Individual, parent2: Individual, 
                                    data: JobShopData) -> Tuple[Individual, Individual]:
        """
        Cruza uniforme poliploide: intercambia genes entre padres para cada cromosoma.
        
        Args:
            parent1: Primer padre
            parent2: Segundo padre
            data: Datos del problema
        
        Returns:
            Dos individuos hijos
        """
        # Crear máscaras aleatorias para el intercambio (una por cromosoma)
        # Cada máscara decide qué genes tomar de cada padre
        child1_chromosomes = {}
        child2_chromosomes = {}
        
        for policy in data.policy_names:
            # Crear máscara aleatoria (True = tomar de parent1, False = tomar de parent2)
            mask = np.random.rand(data.total_operations) < 0.5
            
            # Crear cromosomas de los hijos
            child1_chrom = np.where(mask, parent1.chromosomes[policy], 
                                   parent2.chromosomes[policy])
            child2_chrom = np.where(mask, parent2.chromosomes[policy], 
                                   parent1.chromosomes[policy])
            
            child1_chromosomes[policy] = child1_chrom
            child2_chromosomes[policy] = child2_chrom
        
        # Crear nuevos individuos
        child1 = Individual(data, child1_chromosomes)
        child2 = Individual(data, child2_chromosomes)
        
        return child1, child2
    
    @staticmethod
    def inter_chromosome_mutation(individual: Individual, data: JobShopData):
        """
        Mutación inter-cromosoma: intercambia 2 o 3 cromosomas entre políticas.
        
        Args:
            individual: Individuo a mutar
            data: Datos del problema
        """
        # Seleccionar aleatoriamente 2 o 3 políticas
        num_swaps = random.choice([2, 3])
        policies_to_swap = random.sample(data.policy_names, num_swaps)
        
        # Intercambiar cromosomas de forma circular
        if num_swaps == 2:
            # Intercambio simple entre dos políticas
            p1, p2 = policies_to_swap
            individual.chromosomes[p1], individual.chromosomes[p2] = \
                individual.chromosomes[p2].copy(), individual.chromosomes[p1].copy()
        else:  # num_swaps == 3
            # Intercambio circular: p1 -> p2, p2 -> p3, p3 -> p1
            p1, p2, p3 = policies_to_swap
            temp = individual.chromosomes[p1].copy()
            individual.chromosomes[p1] = individual.chromosomes[p2].copy()
            individual.chromosomes[p2] = individual.chromosomes[p3].copy()
            individual.chromosomes[p3] = temp
        
        # Re-evaluar objetivos
        individual._evaluate()
    
    @staticmethod
    def reciprocal_exchange_mutation(individual: Individual, data: JobShopData, 
                                     num_swaps: int = 2):
        """
        Mutación por intercambio recíproco: intercambia k pares de genes en cada cromosoma.
        
        Args:
            individual: Individuo a mutar
            data: Datos del problema
            num_swaps: Número de pares de genes a intercambiar
        """
        # Para cada cromosoma
        for policy in data.policy_names:
            chromosome = individual.chromosomes[policy]
            
            # Realizar num_swaps intercambios
            for _ in range(num_swaps):
                # Seleccionar dos posiciones aleatorias
                pos1, pos2 = random.sample(range(data.total_operations), 2)
                
                # Intercambiar valores
                chromosome[pos1], chromosome[pos2] = chromosome[pos2], chromosome[pos1]
        
        # Re-evaluar objetivos
        individual._evaluate()
    
    @staticmethod
    def displacement_mutation(individual: Individual, data: JobShopData, 
                             segment_length: int = 3):
        """
        Mutación por desplazamiento: mueve un segmento a otra posición (circular).
        
        Args:
            individual: Individuo a mutar
            data: Datos del problema
            segment_length: Longitud del segmento a desplazar
        """
        # Para cada cromosoma
        for policy in data.policy_names:
            chromosome = individual.chromosomes[policy]
            
            # Seleccionar posición inicial del segmento
            start_pos = random.randint(0, data.total_operations - segment_length)
            
            # Extraer segmento
            segment = chromosome[start_pos:start_pos + segment_length].copy()
            
            # Seleccionar nueva posición (diferente de la actual)
            possible_positions = list(range(0, start_pos)) + \
                               list(range(start_pos + segment_length, 
                                        data.total_operations - segment_length + 1))
            
            if possible_positions:
                new_pos = random.choice(possible_positions)
                
                # Eliminar segmento de posición original
                remaining = np.delete(chromosome, range(start_pos, start_pos + segment_length))
                
                # Insertar en nueva posición
                chromosome = np.insert(remaining, new_pos, segment)
                
                individual.chromosomes[policy] = chromosome
        
        # Re-evaluar objetivos
        individual._evaluate()


# =============================================================================
# ALGORITMO NSGA-II POLIPLOIDE
# =============================================================================

class PolyploidNSGAII:
    """
    Implementación del algoritmo NSGA-II con codificación poliploide.
    """
    
    def __init__(self, data: JobShopData, population_size: int = 20, 
                 generations: int = 100, crossover_rate: float = 0.8,
                 mutation_rates: Dict[str, float] = None, seed: int = None):
        """
        Inicializa el algoritmo NSGA-II poliploide.
        
        Args:
            data: Datos del problema
            population_size: Tamaño de la población
            generations: Número de generaciones
            crossover_rate: Probabilidad de cruza
            mutation_rates: Diccionario con probabilidades de cada mutación
            seed: Semilla para reproducibilidad
        """
        self.data = data
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        
        # Tasas de mutación por defecto
        if mutation_rates is None:
            self.mutation_rates = {
                'inter_chromosome': 0.3,
                'reciprocal_exchange': 0.2,
                'displacement': 0.1
            }
        else:
            self.mutation_rates = mutation_rates
        
        # Establecer semilla si se proporciona
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Inicializar población
        self.population = []
        
        # Estadísticas de mutaciones
        self.mutation_stats = {
            'inter_chromosome': 0,
            'reciprocal_exchange': 0,
            'displacement': 0
        }
        
        # Historial de frentes de Pareto por generación y política
        self.pareto_history = {policy: [] for policy in data.policy_names}
        
        # Historial de hipervolumen
        self.hypervolume_history = {policy: [] for policy in data.policy_names}
    
    def initialize_population(self):
        """
        Crea la población inicial con individuos aleatorios.
        """
        print("Inicializando población...")
        self.population = [Individual(self.data) for _ in range(self.population_size)]
    
    def fast_non_dominated_sort(self, population: List[Individual], policy: str) -> List[List[Individual]]:
        """
        Algoritmo de ordenamiento rápido no-dominado para una política específica.
        
        Args:
            population: Lista de individuos
            policy: Política a evaluar
        
        Returns:
            Lista de frentes (cada frente es una lista de individuos)
        """
        # Inicializar estructuras
        S = [[] for _ in range(len(population))]  # Individuos dominados por i
        n = [0] * len(population)  # Contador de individuos que dominan a i
        rank = [0] * len(population)  # Nivel de no-dominancia
        fronts = [[]]  # Lista de frentes
        
        # Para cada par de individuos
        for i in range(len(population)):
            for j in range(len(population)):
                if i != j:
                    # Si i domina a j
                    if population[i].dominates(population[j], policy):
                        S[i].append(j)
                    # Si j domina a i
                    elif population[j].dominates(population[i], policy):
                        n[i] += 1
            
            # Si nadie domina a i, pertenece al primer frente
            if n[i] == 0:
                rank[i] = 0
                fronts[0].append(i)
        
        # Encontrar frentes subsecuentes
        i = 0
        while i < len(fronts) and fronts[i]:
            next_front = []
            for p_idx in fronts[i]:
                for q_idx in S[p_idx]:
                    n[q_idx] -= 1
                    if n[q_idx] == 0:
                        rank[q_idx] = i + 1
                        next_front.append(q_idx)
            i += 1
            if next_front:
                fronts.append(next_front)
        
        # Convertir índices a individuos
        result_fronts = []
        for front in fronts:
            if front:
                result_fronts.append([population[idx] for idx in front])
        
        return result_fronts
    
    def calculate_crowding_distance(self, front: List[Individual], policy: str):
        """
        Calcula la distancia de crowding para individuos en un frente.
        
        Args:
            front: Lista de individuos en el frente
            policy: Política a evaluar
        """
        if len(front) == 0:
            return
        
        # Inicializar distancias a 0
        for ind in front:
            ind.crowding_distance[policy] = 0
        
        # Si solo hay 1 o 2 individuos, asignar distancia infinita
        if len(front) <= 2:
            for ind in front:
                ind.crowding_distance[policy] = float('inf')
            return
        
        # Para cada objetivo
        for obj_idx in range(2):  # 2 objetivos: makespan y energía
            # Ordenar por objetivo
            front.sort(key=lambda x: x.objectives[policy][obj_idx])
            
            # Asignar distancia infinita a los extremos
            front[0].crowding_distance[policy] = float('inf')
            front[-1].crowding_distance[policy] = float('inf')
            
            # Calcular rango del objetivo
            obj_range = (front[-1].objectives[policy][obj_idx] - 
                        front[0].objectives[policy][obj_idx])
            
            # Evitar división por cero
            if obj_range == 0:
                continue
            
            # Calcular distancia para individuos intermedios
            for i in range(1, len(front) - 1):
                distance = (front[i + 1].objectives[policy][obj_idx] - 
                           front[i - 1].objectives[policy][obj_idx]) / obj_range
                front[i].crowding_distance[policy] += distance
    
    def tournament_selection_with_chromosome_exchange(self, 
                                                     population: List[Individual]) -> Individual:
        """
        Selección por torneo binario con intercambio cromosómico.
        Dos individuos compiten y se recombinan basándose en dominancia y crowding.
        
        Args:
            population: Población actual
        
        Returns:
            Nuevo individuo "super-individuo"
        """
        # Seleccionar dos individuos aleatoriamente
        ind1, ind2 = random.sample(population, 2)
        
        # Crear nuevo individuo seleccionando mejor cromosoma por política
        new_chromosomes = {}
        
        for policy in self.data.policy_names:
            # Comparar basándose en nivel de dominancia y crowding distance
            # (en este contexto usamos los objetivos directamente)
            
            # Determinar mejor cromosoma
            obj1 = ind1.objectives[policy]
            obj2 = ind2.objectives[policy]
            
            # Calcular frentes para determinar nivel de dominancia
            temp_pop = [ind1, ind2]
            fronts = self.fast_non_dominated_sort(temp_pop, policy)
            
            # Asignar ranks
            for rank, front in enumerate(fronts):
                for ind in front:
                    ind.rank = rank
            
            # Calcular crowding distance
            self.calculate_crowding_distance(fronts[0] if fronts else [], policy)
            
            # Seleccionar basándose en rank y crowding distance
            if ind1.rank < ind2.rank:
                selected_chrom = ind1.chromosomes[policy].copy()
            elif ind2.rank < ind1.rank:
                selected_chrom = ind2.chromosomes[policy].copy()
            else:
                # Mismo rank, usar crowding distance
                if (policy in ind1.crowding_distance and 
                    policy in ind2.crowding_distance):
                    if ind1.crowding_distance[policy] > ind2.crowding_distance[policy]:
                        selected_chrom = ind1.chromosomes[policy].copy()
                    else:
                        selected_chrom = ind2.chromosomes[policy].copy()
                else:
                    selected_chrom = random.choice([ind1, ind2]).chromosomes[policy].copy()
            
            new_chromosomes[policy] = selected_chrom
        
        return Individual(self.data, new_chromosomes)
    
    def apply_mutation(self, individual: Individual):
        """
        Aplica aleatoriamente un tipo de mutación al individuo.
        
        Args:
            individual: Individuo a mutar
        """
        # Seleccionar tipo de mutación aleatoriamente según probabilidades
        rand = random.random()
        cumulative = 0
        
        for mutation_type, rate in self.mutation_rates.items():
            cumulative += rate
            if rand < cumulative:
                if mutation_type == 'inter_chromosome':
                    GeneticOperators.inter_chromosome_mutation(individual, self.data)
                    self.mutation_stats['inter_chromosome'] += 1
                elif mutation_type == 'reciprocal_exchange':
                    GeneticOperators.reciprocal_exchange_mutation(individual, self.data)
                    self.mutation_stats['reciprocal_exchange'] += 1
                elif mutation_type == 'displacement':
                    GeneticOperators.displacement_mutation(individual, self.data)
                    self.mutation_stats['displacement'] += 1
                break
    
    def calculate_hypervolume(self, front: List[Individual], policy: str, 
                             reference_point: Tuple[float, float] = None) -> float:
        """
        Calcula el hipervolumen de un frente de Pareto.
        
        Args:
            front: Lista de individuos en el frente
            policy: Política a evaluar
            reference_point: Punto de referencia (si None, se calcula automáticamente)
        
        Returns:
            Valor del hipervolumen
        """
        if len(front) == 0:
            return 0.0
        
        # Obtener objetivos
        objectives = [ind.objectives[policy] for ind in front]
        
        # Si no hay punto de referencia, usar el peor valor + 10%
        if reference_point is None:
            max_makespan = max(obj[0] for obj in objectives)
            max_energy = max(obj[1] for obj in objectives)
            reference_point = (max_makespan * 1.1, max_energy * 1.1)
        
        # Ordenar por primer objetivo
        objectives.sort(key=lambda x: x[0])
        
        # Calcular hipervolumen usando el algoritmo de barrido
        hypervolume = 0.0
        prev_makespan = 0.0
        
        for makespan, energy in objectives:
            if energy < reference_point[1]:
                width = makespan - prev_makespan
                height = reference_point[1] - energy
                hypervolume += width * height
                prev_makespan = makespan
        
        return hypervolume
    
    def run(self, verbose: bool = True) -> List[Individual]:
        """
        Ejecuta el algoritmo NSGA-II poliploide.
        
        Args:
            verbose: Si True, imprime información de progreso
        
        Returns:
            Población final
        """
        # Inicializar población
        self.initialize_population()
        
        # Evolucionar por generaciones
        for gen in range(self.generations):
            if verbose and (gen + 1) % 20 == 0:
                print(f"Generación {gen + 1}/{self.generations}")
            
            # Crear población de descendientes
            offspring = []
            
            while len(offspring) < self.population_size:
                # Selección de padres
                parent1 = self.tournament_selection_with_chromosome_exchange(self.population)
                parent2 = self.tournament_selection_with_chromosome_exchange(self.population)
                
                # Cruza
                if random.random() < self.crossover_rate:
                    child1, child2 = GeneticOperators.uniform_crossover_polyploid(
                        parent1, parent2, self.data)
                else:
                    child1, child2 = parent1, parent2
                
                # Mutación
                total_mutation_prob = sum(self.mutation_rates.values())
                if random.random() < total_mutation_prob:
                    self.apply_mutation(child1)
                if random.random() < total_mutation_prob:
                    self.apply_mutation(child2)
                
                offspring.extend([child1, child2])
            
            # Limitar tamaño de descendientes
            offspring = offspring[:self.population_size]
            
            # Combinar población actual con descendientes
            combined_population = self.population + offspring
            
            # Selección de sobrevivientes por política
            # (Aquí aplicamos el enfoque de super-individuos)
            new_population = []
            
            for _ in range(self.population_size):
                survivor = self.tournament_selection_with_chromosome_exchange(combined_population)
                new_population.append(survivor)
            
            self.population = new_population
            
            # Guardar frentes de Pareto para análisis
            if (gen + 1) in [20, 40, 60, 80, 100]:
                for policy in self.data.policy_names:
                    fronts = self.fast_non_dominated_sort(self.population, policy)
                    if fronts:
                        self.pareto_history[policy].append((gen + 1, fronts[0]))
                        
                        # Calcular hipervolumen
                        hv = self.calculate_hypervolume(fronts[0], policy)
                        self.hypervolume_history[policy].append((gen + 1, hv))
        
        if verbose:
            print("\nEstadísticas de mutaciones:")
            for mut_type, count in self.mutation_stats.items():
                print(f"  {mut_type}: {count}")
        
        return self.population
    
    def get_pareto_front(self, policy: str) -> List[Individual]:
        """
        Obtiene el frente de Pareto de la población final para una política.
        
        Args:
            policy: Nombre de la política
        
        Returns:
            Lista de individuos en el primer frente de Pareto
        """
        fronts = self.fast_non_dominated_sort(self.population, policy)
        return fronts[0] if fronts else []
    
    def find_knee_solution(self, policy: str) -> Individual:
        """
        Encuentra la solución más cercana a la rodilla del frente de Pareto.
        La rodilla es el punto de mejor compromiso entre objetivos.
        
        Args:
            policy: Nombre de la política
        
        Returns:
            Individuo más cercano a la rodilla
        """
        pareto_front = self.get_pareto_front(policy)
        
        if len(pareto_front) == 0:
            return None
        
        # Normalizar objetivos
        makespans = [ind.objectives[policy][0] for ind in pareto_front]
        energies = [ind.objectives[policy][1] for ind in pareto_front]
        
        min_makespan, max_makespan = min(makespans), max(makespans)
        min_energy, max_energy = min(energies), max(energies)
        
        # Evitar división por cero
        if max_makespan == min_makespan:
            makespan_range = 1
        else:
            makespan_range = max_makespan - min_makespan
        
        if max_energy == min_energy:
            energy_range = 1
        else:
            energy_range = max_energy - min_energy
        
        # Calcular distancia al punto ideal normalizado (0, 0)
        best_ind = None
        min_distance = float('inf')
        
        for ind in pareto_front:
            norm_makespan = (ind.objectives[policy][0] - min_makespan) / makespan_range
            norm_energy = (ind.objectives[policy][1] - min_energy) / energy_range
            
            # Distancia euclidiana al origen (punto ideal)
            distance = np.sqrt(norm_makespan**2 + norm_energy**2)
            
            if distance < min_distance:
                min_distance = distance
                best_ind = ind
        
        return best_ind


# =============================================================================
# FUNCIONES DE VISUALIZACIÓN Y REPORTE
# =============================================================================

def plot_pareto_front(algorithm: PolyploidNSGAII, policy: str, 
                      save_path: str = None):
    """
    Grafica el frente de Pareto para una política específica.
    
    Args:
        algorithm: Objeto del algoritmo ejecutado
        policy: Política a graficar
        save_path: Ruta para guardar la imagen (opcional)
    """
    pareto_front = algorithm.get_pareto_front(policy)
    
    if not pareto_front:
        print(f"No hay frente de Pareto para la política {policy}")
        return
    
    # Extraer objetivos
    makespans = [ind.objectives[policy][0] for ind in pareto_front]
    energies = [ind.objectives[policy][1] for ind in pareto_front]
    
    # Crear gráfica
    plt.figure(figsize=(10, 6))
    plt.scatter(makespans, energies, c='blue', marker='o', s=100, alpha=0.6)
    plt.xlabel('Makespan (Tiempo Total)', fontsize=12)
    plt.ylabel('Consumo Energético Total', fontsize=12)
    plt.title(f'Frente de Pareto - Política {policy}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Marcar solución de la rodilla
    knee = algorithm.find_knee_solution(policy)
    if knee:
        knee_makespan, knee_energy = knee.objectives[policy]
        plt.scatter([knee_makespan], [knee_energy], c='red', marker='*', 
                   s=300, label='Solución de la rodilla', zorder=5)
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfica guardada en: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_gantt_chart(individual: Individual, policy: str, 
                      data: JobShopData, save_path: str = None):
    """
    Crea un diagrama de Gantt para visualizar la planificación.
    
    Args:
        individual: Individuo con la solución
        policy: Política utilizada
        data: Datos del problema
        save_path: Ruta para guardar la imagen (opcional)
    """
    # Obtener cromosoma y orden de operaciones
    chromosome = individual.chromosomes[policy]
    operation_order = data.policy_orders[policy]
    
    # Recalcular la planificación con información detallada
    machine_end_times = np.zeros(data.num_machines)
    job_end_times = {j: 0 for j in data.jobs.keys()}
    
    # Almacenar información de cada tarea para el Gantt
    tasks = []  # (machine, start_time, end_time, job_id, operation)
    
    for idx, (job_id, op_idx_in_job) in enumerate(operation_order):
        operation = data.jobs[job_id][op_idx_in_job]
        machine = chromosome[idx] - 1
        
        proc_time = data.processing_times[operation][machine]
        start_time = max(machine_end_times[machine], job_end_times[job_id])
        end_time = start_time + proc_time
        
        tasks.append((machine, start_time, end_time, job_id, operation))
        
        machine_end_times[machine] = end_time
        job_end_times[job_id] = end_time
    
    # Crear gráfica
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Colores por trabajo
    colors = plt.cm.Set3(np.linspace(0, 1, data.num_jobs))
    job_colors = {j: colors[j-1] for j in data.jobs.keys()}
    
    # Dibujar tareas
    for machine, start, end, job_id, operation in tasks:
        duration = end - start
        ax.barh(machine, duration, left=start, height=0.6, 
               color=job_colors[job_id], edgecolor='black', linewidth=0.5)
        
        # Añadir etiqueta
        ax.text(start + duration/2, machine, f'J{job_id}-O{operation+1}', 
               ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Configurar ejes
    ax.set_yticks(range(data.num_machines))
    ax.set_yticklabels([f'M{i+1}' for i in range(data.num_machines)])
    ax.set_xlabel('Tiempo', fontsize=12)
    ax.set_ylabel('Máquinas', fontsize=12)
    ax.set_title(f'Diagrama de Gantt - Política {policy}', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Añadir leyenda
    legend_patches = [mpatches.Patch(color=job_colors[j], label=f'Trabajo {j}') 
                     for j in data.jobs.keys()]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=9)
    
    # Información adicional
    makespan, energy = individual.objectives[policy]
    info_text = f'Makespan: {makespan:.2f}  |  Energía: {energy:.2f}'
    ax.text(0.5, -0.15, info_text, transform=ax.transAxes, 
           ha='center', fontsize=10, bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Diagrama de Gantt guardado en: {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_report(algorithm: PolyploidNSGAII, policy: str, 
                   individual: Individual, filename: str):
    """
    Genera un reporte en formato texto con los resultados.
    
    Args:
        algorithm: Algoritmo ejecutado
        policy: Política del reporte
        individual: Individuo a reportar
        filename: Nombre del archivo de salida
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("REPORTE DE RESULTADOS - PLANIFICACIÓN DE TAREAS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Política utilizada: {policy}\n")
        f.write(f"Makespan: {individual.objectives[policy][0]:.2f}\n")
        f.write(f"Consumo Energético Total: {individual.objectives[policy][1]:.2f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("ASIGNACIÓN DE OPERACIONES\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Posición':<10} {'Trabajo':<10} {'Operación':<12} {'Máquina':<10} {'Tiempo':<10} {'Energía':<10}\n")
        f.write("-" * 80 + "\n")
        
        chromosome = individual.chromosomes[policy]
        operation_order = algorithm.data.policy_orders[policy]
        
        for idx, (job_id, op_idx_in_job) in enumerate(operation_order):
            operation = algorithm.data.jobs[job_id][op_idx_in_job]
            machine = chromosome[idx] - 1
            
            proc_time = algorithm.data.processing_times[operation][machine]
            energy = algorithm.data.energy_consumption[operation][machine]
            
            f.write(f"{idx+1:<10} J{job_id:<9} O{operation+1:<11} M{machine+1:<9} "
                   f"{proc_time:<10.2f} {energy:<10.2f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Reporte generado: {filename}")


# =============================================================================
# FUNCIÓN PRINCIPAL DE EXPERIMENTACIÓN
# =============================================================================

def run_experiment(num_runs: int = 10, generations: int = 100):
    """
    Ejecuta múltiples corridas del algoritmo y recopila estadísticas.
    
    Args:
        num_runs: Número de ejecuciones
        generations: Número de generaciones por ejecución
    """
    print("=" * 80)
    print("INICIO DE EXPERIMENTACIÓN")
    print("=" * 80)
    
    # Inicializar datos del problema
    data = JobShopData()
    
    # Almacenar resultados de hipervolumen
    hypervolume_results = {policy: {gen: [] for gen in [20, 40, 60, 80, 100]} 
                          for policy in data.policy_names}
    
    # Almacenar algoritmos de cada corrida
    algorithms = []
    
    # Ejecutar múltiples corridas
    for run in range(num_runs):
        print(f"\n{'=' * 80}")
        print(f"Ejecución {run + 1}/{num_runs}")
        print(f"{'=' * 80}")
        
        # Crear algoritmo con semilla diferente
        algorithm = PolyploidNSGAII(data, population_size=20, 
                                   generations=generations, seed=run)
        
        # Ejecutar algoritmo
        final_population = algorithm.run(verbose=True)
        
        # Almacenar algoritmo
        algorithms.append(algorithm)
        
        # Recopilar hipervolúmenes
        for policy in data.policy_names:
            for gen, hv in algorithm.hypervolume_history[policy]:
                hypervolume_results[policy][gen].append(hv)
    
    # Calcular estadísticas de hipervolumen
    print("\n" + "=" * 80)
    print("ESTADÍSTICAS DE HIPERVOLUMEN")
    print("=" * 80)
    
    for policy in data.policy_names:
        print(f"\nPolítica: {policy}")
        print(f"{'Generación':<12} {'Min':<12} {'Max':<12} {'Promedio':<12} {'Desv. Est.':<12}")
        print("-" * 60)
        
        for gen in [20, 40, 60, 80, 100]:
            values = hypervolume_results[policy][gen]
            if values:
                min_hv = min(values)
                max_hv = max(values)
                mean_hv = np.mean(values)
                std_hv = np.std(values)
                
                print(f"{gen:<12} {min_hv:<12.2f} {max_hv:<12.2f} "
                      f"{mean_hv:<12.2f} {std_hv:<12.2f}")
    
    # Encontrar la ejecución con desempeño medio
    median_run_idx = num_runs // 2
    median_algorithm = algorithms[median_run_idx]
    
    print(f"\n{'=' * 80}")
    print(f"GENERANDO VISUALIZACIONES (Ejecución con desempeño medio)")
    print(f"{'=' * 80}")
    
    # Para cada política, generar visualizaciones
    for policy in data.policy_names:
        print(f"\nProcesando política: {policy}")
        
        # Graficar frente de Pareto
        plot_pareto_front(median_algorithm, policy, 
                         f"C:/Users/isria/Documents/ESCOM/semestre 8/topicos/practica2/pareto_{policy}.png")
        
        # Obtener solución de la rodilla
        knee_solution = median_algorithm.find_knee_solution(policy)
        
        if knee_solution:
            # Generar diagrama de Gantt para la rodilla
            create_gantt_chart(knee_solution, policy, data, 
                             f"C:/Users/isria/Documents/ESCOM/semestre 8/topicos/practica2/gantt_{policy}_knee.png")
            
            # Generar reporte
            generate_report(median_algorithm, policy, knee_solution, 
                          f"C:/Users/isria/Documents/ESCOM/semestre 8/topicos/practica2/report_{policy}.txt")
        
        # Obtener extremos del frente de Pareto
        pareto_front = median_algorithm.get_pareto_front(policy)
        if len(pareto_front) >= 2:
            # Ordenar por makespan
            pareto_front.sort(key=lambda x: x.objectives[policy][0])
            
            # Extremo con menor makespan
            create_gantt_chart(pareto_front[0], policy, data,
                             f"C:/Users/isria/Documents/ESCOM/semestre 8/topicos/practica2/gantt_{policy}_min_makespan.png")
            
            # Extremo con mayor makespan (menor energía generalmente)
            create_gantt_chart(pareto_front[-1], policy, data,
                             f"C:/Users/isria/Documents/ESCOM/semestre 8/topicos/practica2/gantt_{policy}_min_energy.png")
    
    print(f"\n{'=' * 80}")
    print("EXPERIMENTACIÓN COMPLETADA")
    print(f"{'=' * 80}")
    
    return algorithms, hypervolume_results


# =============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # Ejecutar experimentación completa
    algorithms, hv_results = run_experiment(num_runs=10, generations=100)
    
    print("\n¡Todos los resultados han sido generados exitosamente!")
    print("Revisa la carpeta C:/Users/isria/Documents/ESCOM/semestre 8/topicos/practica2/ para ver los archivos generados.")
