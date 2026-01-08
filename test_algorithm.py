"""
Script de prueba rápida del algoritmo genético poliploide
Ejecuta una versión reducida para verificar que todo funciona correctamente
"""

from polyploid_genetic_algorithm import *

def quick_test():
    """
    Ejecuta una prueba rápida con parámetros reducidos.
    """
    print("="*80)
    print("PRUEBA RÁPIDA DEL ALGORITMO GENÉTICO POLIPLOIDE")
    print("="*80)
    
    # Inicializar datos
    data = JobShopData()
    
    print("\n1. Datos del problema cargados:")
    print(f"   - Número de trabajos: {data.num_jobs}")
    print(f"   - Número de operaciones totales: {data.total_operations}")
    print(f"   - Número de máquinas: {data.num_machines}")
    print(f"   - Políticas: {', '.join(data.policy_names)}")
    
    # Crear y ejecutar algoritmo con parámetros reducidos
    print("\n2. Creando algoritmo NSGA-II...")
    algorithm = PolyploidNSGAII(
        data=data,
        population_size=10,      # Población pequeña para prueba rápida
        generations=20,          # Pocas generaciones para prueba
        crossover_rate=0.8,
        seed=42                  # Semilla fija para reproducibilidad
    )
    
    print("\n3. Ejecutando algoritmo...")
    final_population = algorithm.run(verbose=True)
    
    print(f"\n4. Población final: {len(final_population)} individuos")
    
    # Analizar resultados por política
    print("\n5. Resultados por política:")
    print("-" * 80)
    
    for policy in data.policy_names:
        pareto_front = algorithm.get_pareto_front(policy)
        print(f"\n   Política {policy}:")
        print(f"   - Frente de Pareto: {len(pareto_front)} soluciones")
        
        if pareto_front:
            # Mostrar rango de objetivos
            makespans = [ind.objectives[policy][0] for ind in pareto_front]
            energies = [ind.objectives[policy][1] for ind in pareto_front]
            
            print(f"   - Makespan: [{min(makespans):.2f}, {max(makespans):.2f}]")
            print(f"   - Energía:  [{min(energies):.2f}, {max(energies):.2f}]")
            
            # Encontrar solución de la rodilla
            knee = algorithm.find_knee_solution(policy)
            if knee:
                print(f"   - Solución rodilla: Makespan={knee.objectives[policy][0]:.2f}, "
                      f"Energía={knee.objectives[policy][1]:.2f}")
    
    # Generar algunas visualizaciones
    print("\n6. Generando visualizaciones...")
    
    # Elegir una política para visualizar (FIFO como ejemplo)
    test_policy = 'FIFO'
    
    # Gráfica del frente de Pareto
    plot_pareto_front(algorithm, test_policy, 
                     f"/mnt/user-data/outputs/test_pareto_{test_policy}.png")
    
    # Diagrama de Gantt de la solución de la rodilla
    knee_solution = algorithm.find_knee_solution(test_policy)
    if knee_solution:
        create_gantt_chart(knee_solution, test_policy, data,
                          f"/mnt/user-data/outputs/test_gantt_{test_policy}.png")
        
        # Generar reporte
        generate_report(algorithm, test_policy, knee_solution,
                       f"/mnt/user-data/outputs/test_report_{test_policy}.txt",
                       seed=42)
    
    print("\n" + "="*80)
    print("PRUEBA COMPLETADA EXITOSAMENTE")
    print("="*80)
    print("\nArchivos generados en: /mnt/user-data/outputs/")
    print("- test_pareto_FIFO.png")
    print("- test_gantt_FIFO.png")
    print("- test_report_FIFO.txt")

if __name__ == "__main__":
    quick_test()
