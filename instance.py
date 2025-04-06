import re, numpy as np, random, operator, matplotlib.pyplot as plt
from scipy.spatial.distance import cdist  # used to compute the Euclidean distance matrix
from TSP import *
from CCTPGraph import *

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
	'''
	# Convert city coordinates to numpy format
	positions = dic_to_numpy(cities)

	# Compute Euclidean distance matrix
	distance_matrix = cdist(positions, positions)
	distance_matrix[distance_matrix == 0] = np.inf  # replace diagonals with +inf

	# Create CCTP graph and generate blocked edges
	graph = CCTPGraph(instance, k=10)
	print(f"\n CCTPGraph created with {graph.n} cities")
	print(f" Blocked edges (sample): {list(graph.blocked_edges)[:10]}")

	# Reveal neighboring edges from a specific city (e.g., city 1)
	city = 1
	graph.reveal_at(city)
	blocked_edges = graph.get_blocked_neighbors(city)
	if blocked_edges:
		print(f"City {city} has blocked neighboring edges:")
		for edge in blocked_edges:
			print(f"  - {edge}")
	else:
		print(f"City {city} has no blocked neighboring edges.")

	# Check the known state of specific edges
	sample_check = [(1, 2), (1, 10), (2, 3)]
	print(f"\nEdge state checks:")
	for i, j in sample_check:
		state = "open" if graph.is_open(i, j) else "blocked" if graph.is_blocked(i, j) else "unknown"
		print(f"Edge ({i}, {j}): {state}")

	#ACPM
	graph.christophides_solver() #On appel la fonction christophides
	list_of_edges_selected = graph.christophides.list_of_edges_selected #on recupere la liste de ssommets selectioné dans l'ACPM
	T = graph.christophides.T #On recupere la structure du graphe de l'ACPM
	matching = graph.christophides.matching
	plot_arbre(instance, list_of_edges_selected, cities) #On affiche l'ACPM

	plot_arbre(instance, matching, cities) #On affiche le couplage de cout min du graphe induit
	'''
	graph = CCTPGraph("instances/kroA100.tsp", k=20)
	tsp_path = graph.christophides_solver()
	print("Distance totale (Christofides) :", evaluation(tsp_path, graph.cities))
	plottour(instance, tsp_path, graph.cities, blocked_edges=graph.blocked_edges)