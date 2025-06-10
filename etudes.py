import re,numpy as np, random, operator, matplotlib.pyplot as plt
from instances import *
from TSP import *
from scipy.spatial.distance import cdist  # used to compute the Euclidean distance matrix
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import networkx as nx
import copy
from scipy.sparse.csgraph import dijkstra
from CCTPGraph import *
import random
from collections import deque #algo CR
from CNN import *
from CR import *
from tqdm import tqdm
import time

DEBUG = False

if DEBUG:
    def debug_print(*args, **kwargs):
        print(*args, **kwargs)
else:
    def debug_print(*args, **kwargs):
        pass

# Générer une matrice de villes aléatoires
def generer_matrice_villes(n):

    # Positions aléatoires des villes dans un carré 2D
    positions = np.random.rand(n, 2) * 100  # villes entre (0,0) et (100,100)
    
    # Calcul des distances euclidiennes
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = np.inf  # distance à soi-même = inf
            else:
                mat[i, j] = np.linalg.norm(positions[i] - positions[j])

    return mat, positions






####################################################################################
# Test des algorithmes sur plusieurs tailles de graphes
# et évaluation des performances

def analyse_par_aretes_bloquees(taille_graphe, nb_iterations, k_values):

    christophides_cost = [[0 for _ in range(nb_iterations)] for _ in range(len(k_values))]
    cr_cost = [[0 for _ in range(nb_iterations)] for _ in range(len(k_values))]
    cnn_cost = [[0 for _ in range(nb_iterations)] for _ in range(len(k_values))]
    nb_aretes_bloquees_christophides_path = [[0 for _ in range(nb_iterations)] for _ in range(len(k_values))]

    cr_time = [[0 for _ in range(nb_iterations)] for _ in range(len(k_values))]
    cnn_time = [[0 for _ in range(nb_iterations)] for _ in range(len(k_values))]



    for i in tqdm(range(len(k_values)), desc="k_values"):
        k = k_values[i]  # nombre d'arêtes bloquées
        #print(f"\n==== Taille {k} ====\n")
        
        sommet_depart = 0

        for j in range(nb_iterations):
            #print(f"  Test {j+1}")

            # Générer graphe
            _, positions = generer_matrice_villes(taille_graphe)

            graph = CCTPGraph(cities=positions, k=k) 

            tsp_path = graph.christophides_solver()
            christophides_cost[i][j] = evaluation(tsp_path, graph.cities)


            blocked_on_path = [
                (tsp_path[i], tsp_path[i+1])
                for i in range(len(tsp_path)-1)
                if tuple(sorted((tsp_path[i], tsp_path[i+1]))) in graph.blocked_edges
            ]
            nb_aretes_bloquees_christophides_path[i][j] = len(blocked_on_path)
            
            # Algorithme CNN
            cnn = CNN(graph.distance_matrix, tsp_path, sommet_depart, graph)

            start_cnn = time.perf_counter()
            chemin_final_cnn = cnn.launch()
            end_cnn = time.perf_counter()
            

            cnn_cost[i][j] = evaluation(chemin_final_cnn, graph.cities)
            cnn_time[i][j] = end_cnn - start_cnn  # Temps mis par CNN

            
            # Algorithme CR
            cr_solver = ConstructiveReasoning(graph, tsp_path)

            start_cr = time.perf_counter()
            chemin_final_cr = cr_solver.run()
            end_cr = time.perf_counter()

            cr_cost[i][j] = evaluation(chemin_final_cr, graph.cities)
            cr_time[i][j] = end_cr - start_cr  # Temps mis par CR

    # Calcul de la moyenne et écart-type
    cr_cost = np.array(cr_cost)
    cnn_cost = np.array(cnn_cost)
    cr_time = np.array(cr_time)
    cnn_time = np.array(cnn_time)

    cr_mean = cr_cost.mean(axis=1)
    cr_std = cr_cost.std(axis=1)

    cnn_mean = cnn_cost.mean(axis=1)
    cnn_std = cnn_cost.std(axis=1)

    cr_time_mean = cr_time.mean(axis=1)
    cr_time_std = cr_time.std(axis=1)

    cnn_time_mean = cnn_time.mean(axis=1)
    cnn_time_std = cnn_time.std(axis=1)

    # Tracer
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, cr_mean, '-o', label='CR')
    plt.fill_between(k_values, cr_mean - cr_std, cr_mean + cr_std, alpha=0.3)

    plt.plot(k_values, cnn_mean, '-s', label='CNN')
    plt.fill_between(k_values, cnn_mean - cnn_std, cnn_mean + cnn_std, alpha=0.3)

    #plt.errorbar(k_values, cr_mean, yerr=cr_std, label='CR', fmt='-o', capsize=5) # CR
    #plt.errorbar(k_values, cnn_mean, yerr=cnn_std, label='CNN', fmt='-s', capsize=5) # CNN

    # Mise en forme
    plt.xlabel('Nombre d\'arêtes bloquées')
    plt.ylabel('Coût moyen')
    plt.title('Comparaison des algorithmes CR et CNN')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Afficher
    plt.show()


    # Graphique Temps d'exécution
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, cr_time_mean, '-o', label='CR - Temps moyen')
    plt.fill_between(k_values, cr_time_mean - cr_time_std, cr_time_mean + cr_time_std, alpha=0.3)

    plt.plot(k_values, cnn_time_mean, '-s', label='CNN - Temps moyen')
    plt.fill_between(k_values, cnn_time_mean - cnn_time_std, cnn_time_mean + cnn_time_std, alpha=0.3)

    plt.xlabel('Nombre d\'arêtes bloquées')
    plt.ylabel('Temps moyen d\'exécution (secondes)')
    plt.title('Temps d\'exécution moyen des algorithmes CR et CNN')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analyse_par_taille_graphes(taille_graphes, nb_iterations):
    christophides_cost = [[0 for _ in range(nb_iterations)] for _ in range(len(taille_graphes))]
    cr_cost = [[0 for _ in range(nb_iterations)] for _ in range(len(taille_graphes))]
    cnn_cost = [[0 for _ in range(nb_iterations)] for _ in range(len(taille_graphes))]
    nb_aretes_bloquees_christophides_path = [[0 for _ in range(nb_iterations)] for _ in range(len(taille_graphes))]

    cr_time = [[0 for _ in range(nb_iterations)] for _ in range(len(taille_graphes))]
    cnn_time = [[0 for _ in range(nb_iterations)] for _ in range(len(taille_graphes))]


    for i in tqdm(range(len(taille_graphes)), desc="Taille de graphe"):
        k = taille_graphes[i]-2  # nombre d'arêtes bloquées
        debug_print(f"\n==== Taille {taille_graphes[i]} ====\n")
                
        sommet_depart = 0

        for j in range(nb_iterations):
            debug_print(f"  Test {j+1}")

            # Générer graphe
            _, positions = generer_matrice_villes(taille_graphes[i])

            graph = CCTPGraph(cities=positions, k=k) 

            tsp_path = graph.christophides_solver()
            christophides_cost[i][j] = evaluation(tsp_path, graph.cities)


            blocked_on_path = [
                (tsp_path[i], tsp_path[i+1])
                for i in range(len(tsp_path)-1)
                if tuple(sorted((tsp_path[i], tsp_path[i+1]))) in graph.blocked_edges
            ]
            nb_aretes_bloquees_christophides_path[i][j] = len(blocked_on_path)
            
            # Algorithme CNN
            cnn = CNN(graph.distance_matrix, tsp_path, sommet_depart, graph)
            start_cnn = time.perf_counter()
            chemin_final_cnn = cnn.launch()
            end_cnn = time.perf_counter()
            

            cnn_cost[i][j] = evaluation(chemin_final_cnn, graph.cities)
            cnn_time[i][j] = end_cnn - start_cnn  # Temps mis par CNN
            
            # Algorithme CR
            cr_solver = ConstructiveReasoning(graph, tsp_path)

            start_cr = time.perf_counter()
            chemin_final_cr = cr_solver.run()
            end_cr = time.perf_counter()

            cr_cost[i][j] = evaluation(chemin_final_cr, graph.cities)
            cr_time[i][j] = end_cr - start_cr  # Temps mis par CR

    # Calcul de la moyenne et écart-type
    cr_cost = np.array(cr_cost)
    cnn_cost = np.array(cnn_cost)
    cr_time = np.array(cr_time)
    cnn_time = np.array(cnn_time)

    cr_mean = cr_cost.mean(axis=1)
    cr_std = cr_cost.std(axis=1)

    cnn_mean = cnn_cost.mean(axis=1)
    cnn_std = cnn_cost.std(axis=1)

    cr_time_mean = cr_time.mean(axis=1)
    cr_time_std = cr_time.std(axis=1)

    cnn_time_mean = cnn_time.mean(axis=1)
    cnn_time_std = cnn_time.std(axis=1)

    # Tracer
    plt.figure(figsize=(10, 6))
    plt.plot(taille_graphes, cr_mean, '-o', label='CR')
    plt.fill_between(taille_graphes, cr_mean - cr_std, cr_mean + cr_std, alpha=0.3)


    plt.plot(taille_graphes, cnn_mean, '-s', label='CNN')
    plt.fill_between(taille_graphes, cnn_mean - cnn_std, cnn_mean + cnn_std, alpha=0.3)

    #plt.errorbar(taille_graphes, cr_mean, yerr=cr_std, label='CR', fmt='-o', capsize=5) # CR
    #plt.errorbar(taille_graphes, cnn_mean, yerr=cnn_std, label='CNN', fmt='-s', capsize=5) # CNN

    # Mise en forme
    plt.xlabel('Taille du graphe (nombre de sommets)')
    plt.ylabel('Coût moyen')
    plt.title('Comparaison des algorithmes CR et CNN')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Afficher
    plt.show()


    # Graphique Temps d'exécution
    plt.figure(figsize=(10, 6))
    plt.plot(taille_graphes, cr_time_mean, '-o', label='CR - Temps moyen')
    plt.fill_between(taille_graphes, cr_time_mean - cr_time_std, cr_time_mean + cr_time_std, alpha=0.3)

    plt.plot(taille_graphes, cnn_time_mean, '-s', label='CNN - Temps moyen')
    plt.fill_between(taille_graphes, cnn_time_mean - cnn_time_std, cnn_time_mean + cnn_time_std, alpha=0.3)

    plt.xlabel('Taille du graphe (nombre de sommets)')
    plt.ylabel('Temps moyen d\'exécution (secondes)')
    plt.title('Temps d\'exécution moyen des algorithmes CR et CNN')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    taille_graphes = np.linspace(150, 350, 50, dtype=int)
    nb_iterations = 5
    analyse_par_taille_graphes(taille_graphes, nb_iterations)

    taille_graphe = 200
    k_values = np.linspace(0, taille_graphe-2, 50, dtype=int)
    analyse_par_aretes_bloquees(taille_graphe, nb_iterations, k_values)

