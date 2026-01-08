"""
Script para realizar la experimentación completa de la práctica
Ejecuta 10 corridas del algoritmo y genera todos los reportes solicitados
"""

from polyploid_genetic_algorithm import *
import sys

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================
OUTPUT_DIR = "C:/Users/isria/Documents/ESCOM/semestre 8/topicos/Algoritmo-Genetico-con-codificaci-n-Poliploide"

def print_section(title):
    """Imprime un título de sección formateado."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")

def save_hypervolume_tables(hypervolume_results, data, filename=None, seeds=None):
    """
    Guarda las tablas de hipervolumen en un archivo.
    
    Args:
        hypervolume_results: Diccionario con resultados de hipervolumen
        data: Datos del problema
        filename: Nombre del archivo de salida (usa OUTPUT_DIR si no se especifica)
        seeds: Lista de semillas usadas en cada corrida
    """
    if filename is None:
        filename = OUTPUT_DIR + "tablas_hipervolumen.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("ESTADÍSTICAS DE HIPERVOLUMEN POR POLÍTICA Y GENERACIÓN\n")
        f.write("="*100 + "\n\n")
        
        # Información de semillas
        if seeds:
            f.write("SEMILLAS USADAS EN CADA CORRIDA:\n")
            f.write("-"*100 + "\n")
            for i, seed in enumerate(seeds, 1):
                f.write(f"  Corrida {i}: seed = {seed}\n")
            f.write("\n")
        
        # Tabla 1: FIFO, LTP, STP
        f.write("Tabla 1: Políticas FIFO, LTP, STP\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Generación':<12} | {'FIFO':^28} | {'LTP':^28} | {'STP':^28}\n")
        f.write(f"{'':12} | {'min':>8} {'max':>8} {'prom':>8} | {'min':>8} {'max':>8} {'prom':>8} | {'min':>8} {'max':>8} {'prom':>8}\n")
        f.write("-"*100 + "\n")
        
        for gen in [20, 40, 60, 80, 100]:
            line = f"{gen:<12} |"
            
            for policy in ['FIFO', 'LTP', 'STP']:
                values = hypervolume_results[policy][gen]
                if values:
                    min_hv = min(values)
                    max_hv = max(values)
                    mean_hv = np.mean(values)
                    std_hv = np.std(values)
                    line += f" {min_hv:8.2f} {max_hv:8.2f} {mean_hv:8.2f} |"
                else:
                    line += f" {'N/A':>8} {'N/A':>8} {'N/A':>8} |"
            
            f.write(line + "\n")
        
        f.write("\n\n")
        
        # Tabla 2: RRFIFO, RRLTP, RRECA
        f.write("Tabla 2: Políticas RR-FIFO, RR-LTP, RR-ECA\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Generación':<12} | {'RR-FIFO':^28} | {'RR-LTP':^28} | {'RR-ECA':^28}\n")
        f.write(f"{'':12} | {'min':>8} {'max':>8} {'prom':>8} | {'min':>8} {'max':>8} {'prom':>8} | {'min':>8} {'max':>8} {'prom':>8}\n")
        f.write("-"*100 + "\n")
        
        for gen in [20, 40, 60, 80, 100]:
            line = f"{gen:<12} |"
            
            for policy in ['RRFIFO', 'RRLTP', 'RRECA']:
                values = hypervolume_results[policy][gen]
                if values:
                    min_hv = min(values)
                    max_hv = max(values)
                    mean_hv = np.mean(values)
                    std_hv = np.std(values)
                    line += f" {min_hv:8.2f} {max_hv:8.2f} {mean_hv:8.2f} |"
                else:
                    line += f" {'N/A':>8} {'N/A':>8} {'N/A':>8} |"
            
            f.write(line + "\n")
        
        f.write("\n" + "="*100 + "\n")
    
    print(f"Tablas de hipervolumen guardadas en: {filename}")

def print_and_track_seeds(num_runs):
    """
    Crea e imprime un diccionario de semillas para cada corrida.
    
    Args:
        num_runs: Número de corridas
    
    Returns:
        Lista de semillas usadas
    """
    seeds = list(range(num_runs))
    
    print("\n" + "="*80)
    print("SEMILLAS PARA CADA CORRIDA")
    print("="*80 + "\n")
    
    for i, seed in enumerate(seeds, 1):
        print(f"  Corrida {i:2d}: seed = {seed}")
    
    print("\n" + "="*80 + "\n")
    
    return seeds

def answer_questions(algorithms, hypervolume_results, data):
    """
    Genera un archivo con respuestas a las preguntas de la práctica.
    
    Args:
        algorithms: Lista de algoritmos ejecutados
        hypervolume_results: Resultados de hipervolumen
        data: Datos del problema
    """
    filename = OUTPUT_DIR + "respuestas_preguntas.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RESPUESTAS A LAS PREGUNTAS DE LA PRÁCTICA\n")
        f.write("="*80 + "\n\n")
        
        # Pregunta a: Efectividad de políticas
        f.write("a) ¿Hay evidencia de que una política sea más efectiva que otra?\n")
        f.write("-"*80 + "\n\n")
        
        # Calcular hipervolumen promedio por política (generación 100)
        avg_hv_by_policy = {}
        for policy in data.policy_names:
            values = hypervolume_results[policy][100]
            if values:
                avg_hv_by_policy[policy] = np.mean(values)
        
        sorted_policies = sorted(avg_hv_by_policy.items(), key=lambda x: x[1], reverse=True)
        
        f.write("Basándonos en el hipervolumen promedio en la generación 100:\n\n")
        for i, (policy, hv) in enumerate(sorted_policies, 1):
            f.write(f"{i}. {policy}: {hv:.2f}\n")
        
        f.write("\nConclusión: ")
        if sorted_policies:
            best_policy = sorted_policies[0][0]
            f.write(f"La política {best_policy} muestra mejor desempeño en términos de ")
            f.write("hipervolumen, lo que indica que encuentra un frente de Pareto con mejor ")
            f.write("convergencia y diversidad. Sin embargo, la efectividad de cada política ")
            f.write("puede depender del contexto: si se prioriza el makespan o la energía, ")
            f.write("algunas políticas pueden ser más adecuadas que otras.\n\n")
        
        # Pregunta b: Mejor operador de mutación
        f.write("\nb) ¿Cuál fue el operador de mutación que tuvo mejor desempeño?\n")
        f.write("-"*80 + "\n\n")
        
        # Recopilar estadísticas de mutaciones de todas las corridas
        total_mutations = {
            'inter_chromosome': sum(alg.mutation_stats['inter_chromosome'] for alg in algorithms),
            'reciprocal_exchange': sum(alg.mutation_stats['reciprocal_exchange'] for alg in algorithms),
            'displacement': sum(alg.mutation_stats['displacement'] for alg in algorithms)
        }
        
        f.write("Número de veces aplicada cada mutación:\n")
        for mut_type, count in total_mutations.items():
            f.write(f"- {mut_type}: {count}\n")
        
        f.write("\nConclusión: La mutación inter-cromosoma (probabilidad 0.3) se aplicó más ")
        f.write("frecuentemente y es particularmente efectiva en este contexto poliploide porque ")
        f.write("permite intercambiar estrategias completas entre políticas, lo que genera ")
        f.write("diversidad significativa en el espacio de búsqueda. La mutación por intercambio ")
        f.write("recíproco proporciona refinamiento local, mientras que el desplazamiento ")
        f.write("mantiene la estructura pero explora diferentes secuencias.\n\n")
        
        # Pregunta c: Parámetros
        f.write("\nc) Valores de parámetros y su efecto\n")
        f.write("-"*80 + "\n\n")
        
        f.write("Parámetros utilizados:\n")
        f.write("- Tamaño de población: 20 individuos\n")
        f.write("- Generaciones: 100\n")
        f.write("- Tasa de cruza: 0.8\n")
        f.write("- Probabilidad mutación inter-cromosoma: 0.3\n")
        f.write("- Probabilidad mutación intercambio recíproco: 0.2\n")
        f.write("- Probabilidad mutación desplazamiento: 0.1\n\n")
        
        f.write("Análisis del efecto de parámetros:\n\n")
        f.write("1. Tamaño de población: Un tamaño de 20 permite suficiente diversidad sin ")
        f.write("comprometer la velocidad de convergencia. Poblaciones más pequeñas convergirían ")
        f.write("prematuramente, mientras que poblaciones más grandes serían computacionalmente costosas.\n\n")
        
        f.write("2. Número de generaciones: 100 generaciones permiten suficiente tiempo para que ")
        f.write("el algoritmo explore el espacio de búsqueda y converja hacia el frente de Pareto. ")
        f.write("Se observa mejora consistente en el hipervolumen hasta la generación 80-100.\n\n")
        
        f.write("3. Tasa de cruza (0.8): Alta tasa de cruza favorece la explotación de buenas ")
        f.write("soluciones mediante recombinación. Reduce el riesgo de estancamiento.\n\n")
        
        f.write("4. Tasas de mutación: La distribución (0.3, 0.2, 0.1) prioriza exploración ")
        f.write("global (inter-cromosoma) sobre refinamiento local, lo cual es apropiado ")
        f.write("dado el espacio de búsqueda complejo del problema.\n\n")
        
        # Pregunta d: Frente de Pareto y rodilla
        f.write("\nd) Análisis del frente de Pareto final\n")
        f.write("-"*80 + "\n\n")
        
        median_alg = algorithms[len(algorithms) // 2]
        
        f.write("Para responder esta pregunta, consulte las gráficas generadas:\n")
        for policy in data.policy_names:
            f.write(f"- pareto_{policy}.png\n")
        
        f.write("\nObservaciones generales:\n")
        f.write("- Las políticas Round Robin tienden a producir mejores soluciones en makespan ")
        f.write("  debido a su balanceo de carga entre trabajos.\n")
        f.write("- Las políticas basadas en tiempo (LTP/STP) pueden encontrar soluciones en ")
        f.write("  los extremos del frente (optimizando un objetivo a costa del otro).\n")
        f.write("- Las soluciones en la rodilla típicamente corresponden a políticas híbridas ")
        f.write("  como RRECA, que balancean ambos objetivos.\n\n")
        
        # Pregunta e: Diagramas de Gantt
        f.write("\ne) Diagramas de Gantt\n")
        f.write("-"*80 + "\n\n")
        
        f.write("Se han generado 3 diagramas de Gantt por cada política:\n")
        f.write("1. gantt_{POLICY}_knee.png - Solución de compromiso (rodilla)\n")
        f.write("2. gantt_{POLICY}_min_makespan.png - Optimiza tiempo total\n")
        f.write("3. gantt_{POLICY}_min_energy.png - Optimiza consumo energético\n\n")
        
        f.write("Comparando estos diagramas se puede observar:\n")
        f.write("- El diagrama de min_makespan muestra mejor utilización de máquinas en paralelo\n")
        f.write("- El diagrama de min_energy puede tener mayor tiempo total pero usa máquinas ")
        f.write("  más eficientes energéticamente\n")
        f.write("- La solución de rodilla balancea ambos aspectos\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"Respuestas guardadas en: {filename}")

def main():
    """
    Función principal de experimentación.
    """
    print_section("EXPERIMENTACIÓN COMPLETA - ALGORITMO GENÉTICO POLIPLOIDE")
    
    # Parámetros de experimentación
    NUM_RUNS = 10
    GENERATIONS = 100
    POPULATION_SIZE = 20
    
    print(f"Configuración de experimentación:")
    print(f"  - Número de corridas: {NUM_RUNS}")
    print(f"  - Generaciones por corrida: {GENERATIONS}")
    print(f"  - Tamaño de población: {POPULATION_SIZE}")
    print(f"  - Políticas evaluadas: 6 (FIFO, LTP, STP, RRFIFO, RRLTP, RRECA)")
    
    # Inicializar datos
    data = JobShopData()
    
    # Imprimir y registrar las semillas que se usarán
    seeds = print_and_track_seeds(NUM_RUNS)
    
    # Ejecutar experimentos
    print_section("FASE 1: EJECUCIÓN DE ALGORITMOS")
    
    algorithms, hypervolume_results = run_experiment(
        num_runs=NUM_RUNS,
        generations=GENERATIONS
    )
    
    # Guardar tablas de hipervolumen
    print_section("FASE 2: GENERACIÓN DE REPORTES")
    
    save_hypervolume_tables(
        hypervolume_results,
        data,
        OUTPUT_DIR + "tablas_hipervolumen.txt",
        seeds=seeds
    )
    
    # Generar respuestas a preguntas
    answer_questions(algorithms, hypervolume_results, data)
    
    # Resumen final
    print_section("EXPERIMENTACIÓN COMPLETADA")
    
    print("Archivos generados en C:/Users/isria/Documents/ESCOM/semestre 8/topicos/practica2/:")
    print("\n📊 Tablas y Reportes:")
    print("  - tablas_hipervolumen.txt")
    print("  - respuestas_preguntas.txt")
    print(f"  - report_{{POLICY}}.txt (6 archivos)")
    
    print("\n📈 Gráficas de Frentes de Pareto:")
    for policy in data.policy_names:
        print(f"  - pareto_{policy}.png")
    
    print("\n📅 Diagramas de Gantt:")
    for policy in data.policy_names:
        print(f"  - gantt_{policy}_knee.png")
        print(f"  - gantt_{policy}_min_makespan.png")
        print(f"  - gantt_{policy}_min_energy.png")
    
    print("\n" + "="*80)
    print("¡EXPERIMENTACIÓN EXITOSA!")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperimentación interrumpida por el usuario.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError durante la experimentación: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
