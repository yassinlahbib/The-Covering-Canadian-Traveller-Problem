import re,numpy as np, random, operator, matplotlib.pyplot as plt
from instances import *
from TSP import *
from scipy.spatial.distance import cdist  # used to compute the Euclidean distance matrix
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import networkx as nx
import copy
from scipy.sparse.csgraph import dijkstra

class CNN():
    def __init__(self, distance_matrix, tsp_path, original_cities, cctp_graph):
        self.distance_matrix = distance_matrix #matrice de distance
        self.P = tsp_path #chemin retourné par Christophides
        self.s = original_cities #ville d'origine
        self.cctp_graph = cctp_graph #structure du graphe de l'instance
        self.n = len(self.distance_matrix) #nombre de villes
        print("DEBUT")
        print(f"{self.distance_matrix=}")

    def launch(self):
        Gstar, Us, P_prime = self.shortcut()
        G_prime = self.compress(Us=Us, P_prime=P_prime) 
        p_2 = self.algo_exploration(G_prime)
        print("Chemin Final=", P_prime + p_2)
        return P_prime + p_2


    def blocked_edges(self, i):
        """ Renvoie les arêtes bloquées incidentes à i, et met à jour la distance_matrix """
        blocked = set()
        for j in range (self.n):
            if tuple(sorted((i, j))) in self.cctp_graph.blocked_edges:
                blocked.add(tuple(sorted((i, j))))
                print(f"Blocked edge: {i} -> {j}")
                self.distance_matrix[i][j] = np.inf
                self.distance_matrix[j][i] = np.inf
        return blocked

    def shortcut(self):
        i = 0
        j = 1
        Eb = set() # ensemble des arêtes bloquées
        Us = {self.s} # ensemble des sommets non visités
        P_prime = [self.s] #chemin P' que le voyageur construit

        while j < self.n :
            Eb = Eb | self.blocked_edges(self.P[i]) # On ajoute les arêtes bloquées incidentes à i
            print(f"P[i]={self.P[i]} -> Blocked edges: {self.blocked_edges(self.P[i])}")
            if (self.P[i], self.P[j]) not in Eb:
                P_prime.append(self.P[j]) # On ajoute la ville j à P'
                i = j
            else :
                Us.add(self.P[j]) # On ajoute la ville j à Us
                print(f"Blocked edge: {Us=}")
            j = j+1
        print(f"nombre de blocked edges: {len(Eb)}")
        if tuple(sorted((self.P[i], self.s))) in Eb:
            print("ici")
            P_prime = P_prime + P_prime[::-1][1:] # On ajoute le chemin inverse sans répéter la derniere ville du chemin P'
        else :
            P_prime.append(self.s)

        #On a remplacé Gstar le nouveaau graphe sans arêtes bloquées par la matrice de distance
        Gstar = self.distance_matrix #(i,j) = np.inf pour les arêtes bloquées découvertes

        print(f"{Gstar=}")
        print(f"{Us=}")
        print(f"{P_prime=}")
        return Gstar, Us, P_prime

    def compress(self, Us, P_prime):
        
        print("AVANT")
        print(f"{self.distance_matrix=}")
        # GRAPHE G'
        G_prime = copy.deepcopy(self.distance_matrix) #Graphe (Us, E') sommet non visité(Us) et arêtes entre les sommets de Us
        for i in range (self.n):
            if i not in Us:
                print(f"i={i} not in {Us}")
                G_prime[i, :] = np.inf #On enleve les sommets visités
                G_prime[:, i] = np.inf 
        print("APRES")
        print(f"{G_prime=}")

        # GRAPHE H
        H = copy.deepcopy(self.distance_matrix) #Graphe (V, E\E') les aretes tq on connait leurs état
        #On enlève les arêtes bloquées
        for i in range (self.n):
            for j in range (self.n):
                if (i!=j) and (i in Us) and (j in Us): #Si (i,j) in E'. Avec E' = {(i,j) | i,j in Us}
                    H[i][j] = np.inf
        print(f"{H=}")

        
        self.shortest_path = {}
        Us = list(Us) #On convertit en liste pour l'indexation
        for i in range (len(Us)):
            for j in range (i+1, len(Us)):
                #Trouver plus court chemin entre i et j dans H
                distances, predecesseurs = dijkstra(H, return_predecessors=True, indices=Us[i]) #On lance Dijkstra à partir du sommet Us[i] dans H 
                #On recupere le cout du chemin entre i et j
                cout = distances[Us[j]]
                if G_prime[Us[i]][Us[j]] > cout:
                    G_prime[Us[i]][Us[j]] = cout
                    G_prime[Us[j]][Us[i]] = cout
                    #On recupere le chemin entre i et j
                    chemin = []
                    current = Us[j]
                    while current != Us[i]:
                        chemin.append(current)
                        current = predecesseurs[current]
                    chemin.append(Us[i])
                    chemin.reverse()
                    print(f"Chemin entre {Us[i]} et {Us[j]}: {chemin}")
                    self.shortest_path[Us[i], Us[j]] = chemin
                    self.shortest_path[Us[j], Us[i]] = chemin[::-1] #On ajoute le chemin inverse (de Us[j] à Us[i]) 
                    
        print(f"{self.shortest_path=}")

        
        print(f"{G_prime=}")
        return G_prime
                
        
    def algo_exploration(self, G):
        print("ALGO EXPLORATION")
        path = []

        current = self.s
        to_initial_state = copy.deepcopy(G[self.s]) #On garde l'état initial du sommet de départ pour revenir à l'état initial

        while True:
            print(f"G={G}")
            print(f"current={current}")
            if np.all(G == np.inf):
                print("Aucun sommet n'est atteignable")
                if current == self.s:
                    print("On à fini d'explorer tous les sommets avant d'entrer dans la fonction, c'est a dire Us contient seulement la ville de depart", )
                    return path
                break
            # On cherche le voisin le plus proche atteignable
            voisin = np.argmin(G[current]) #On recupere le sommet le plus proche de chaque sommet
            print(f"voisin={voisin}")
            if (current, voisin) in self.shortest_path:
                print(f"chemin entre {current} et {voisin} : {self.shortest_path[current, voisin][1:]} de distance {G[current][voisin]}")
                path+=self.shortest_path[current, voisin][1:] #utilisation du plus court chemin trouvé dans COMPRESS
            else:
                path.append(voisin) #arete directe
                print(f"arete directe entre {current} et {voisin} de distance {G[current][voisin]}")
            G[current,:] = np.inf #On enlève le sommet courant pour ne pas le visiter à nouveau
            G[:,current] = np.inf
            current = voisin
            
            if current == self.s:
                print("On est revenu au sommet de départ")
                break
        print("On a fini d'explorer tous les sommets")
        print(f"Chemin exploré: {path}")
        #On retourne au sommet de départ
        current = path[-1]
        if self.shortest_path.get((current, self.s)) is not None:
            path += self.shortest_path[current, self.s][1:]
            print(f"chemin entre {current} et {self.s} : {self.shortest_path[current, self.s][1:]}")
        else:
            path.append(self.s)
            print(f"arete directe entre {current} et {self.s} de distance {to_initial_state[current]}")
        print(f"Chemin exploré: {path}")
        return path
    

import re, numpy as np, random, operator, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist  # used to compute the Euclidean distance matrix
from TSP import *
from CCTPGraph import *
from CNN import *
import random

#random.seed(42)

def dic_to_numpy(dic):
	""" Convert dictionary of positions to a numpy array """
	res = []
	for key in dic:
		res.append(dic[key])
	return np.array(res)



if __name__ == "__main__":
	
	instance = "instances/kroA100.tsp"
	data = read_tsp_data(instance)
	nbCities = int(detect_dimension(data))
	cities = read_tsp(nbCities, data)
	print(f"Number of cities: {nbCities}")

	graph = CCTPGraph("instances/kroA100.tsp", k=8)
	tsp_path = graph.christophides_solver()
	print("Distance totale (Christofides) :", evaluation(tsp_path, graph.cities))
	print("Chemin TSP (Christofides) :", tsp_path)

	print("\n\nTest CNN Algorithme\n\n")
	cnn = CNN(graph.distance_matrix, tsp_path, 0, graph)
	chemin_final = cnn.launch() 
	plottour(instance, chemin_final, graph.cities, blocked_edges=graph.blocked_edges)
	print("Distance totale (CNN) :", evaluation(chemin_final, graph.cities))
	print("Chemin final (CNN) :", chemin_final)