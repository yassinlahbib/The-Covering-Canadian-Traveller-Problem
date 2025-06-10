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
	#plottour(instance, tsp_path, graph.cities, blocked_edges=graph.blocked_edges)

	print("\n\nTest CNN Algorithme\n\n")
	cnn = CNN(graph.distance_matrix, tsp_path, 0, graph)
	chemin_final = cnn.launch() 
	plottour(instance, chemin_final, graph.cities, blocked_edges=graph.blocked_edges)
	print("Distance totale (CNN) :", evaluation(chemin_final, graph.cities))
	print("Chemin final (CNN) :", chemin_final)