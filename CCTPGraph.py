import re,numpy as np, random, operator, matplotlib.pyplot as plt
from instances import *
from TSP import *
from scipy.spatial.distance import cdist  # used to compute the Euclidean distance matrix
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import networkx as nx


class CCTPGraph:
	def __init__(self, tsp_file, k=10):
		self.tsp_file = tsp_file
		self.k = k
		self.cities = self._load_cities()
		self.n = len(self.cities)
		self.blocked_edges = self._generate_blocked_edges()
		print(f"{self.k} blocked edges generated: {self.blocked_edges}")
		self.known_edges = {}  # (i,j) → 'open' or 'blocked'

		cities_numpy = self.dic_to_numpy(self.cities)
		self.distance_matrix = cdist(cities_numpy, cities_numpy) # Compute Euclidean distance matrix (i,j) corresponding cost of edge. because complete graph
		self.distance_matrix[self.distance_matrix == 0] = np.inf  # replace diagonals with +inf

		# J'ai mis les lignes suivants en commentaires car Christophidesn'est pas censé prendre en compte les arêtes bloquées
		# # Appliquer les arêtes bloquées 
		# for i, j in self.blocked_edges:
		# 	self.distance_matrix[i][j] = np.inf
		# 	self.distance_matrix[j][i] = np.inf

		assert (self.k < self.n -1) #on doit avoir au plus k blocages avec k < n-1

	def dic_to_numpy(self, dic):
		""" Convert dictionary of positions to a numpy array """
		res = []
		for key in dic:
			res.append(dic[key])
		return np.array(res)

	def _load_cities(self):
		data = read_tsp_data(self.tsp_file)
		dimension = int(detect_dimension(data))
		return read_tsp(dimension, data)

	def _generate_blocked_edges(self):
		all_edges = [(i, j) for i in self.cities for j in self.cities if i < j]
		if self.k >= len(all_edges):
			raise ValueError("Too many blocked edges; the complete graph would become disconnected.")
		return set(random.sample(all_edges, self.k))

	def reveal_at(self, node):
		for neighbor in self.cities:
			if node == neighbor:
				continue
			edge = tuple(sorted((node, neighbor)))
			if edge not in self.known_edges:
				if edge in self.blocked_edges:
					self.known_edges[edge] = 'blocked'
				else:
					self.known_edges[edge] = 'open'

	def get_blocked_neighbors(self, node):
		blocked = []
		for neighbor in self.cities:
			if neighbor == node:
				continue
			edge = tuple(sorted((node, neighbor)))
			if self.is_blocked(*edge):
				blocked.append(edge)
		return blocked

	def is_known(self, i, j):
		edge = tuple(sorted((i, j)))
		return edge in self.known_edges

	def is_blocked(self, i, j):
		edge = tuple(sorted((i, j)))
		print(f"{self.known_edges.get(edge)}")
		return self.known_edges.get(edge) == 'blocked'

	def is_open(self, i, j):
		edge = tuple(sorted((i, j)))
		return self.known_edges.get(edge) == 'open'
	
	def christophides_solver(self):
		""" applique l'algo de christophides sur notre instance """
		self.christophides = Christophides(self.distance_matrix)
		return self.christophides.launch()

	
#Pour le moment je n'ai pas pris en compte les blocages, dans un premier temps je me concentre principalemnt sur l'algo de cgristophides
class Christophides():
	""" Applique algo de christophides sur graphe complet respectant l'inégalité triangulaire """
	def __init__(self, distance_matrix):
		self.distance_matrix = distance_matrix

	# Arbre couvrant de poids minimum 
	def ACPM(self, verbose=False):
		"""
			Renvoies les aretes d'un arbre couvrant de cout min
			-> Basé sur Algorithme de Prim
		Args: 
			distance_matrix: np.array
				matrice des distances entre chaque sommet, vaut +inf en diagonale

		Returns:
			list_of_edges_selected : list[(u,v)]
				liste des aretes séléctionnées pour l'ACPM
			T : dic(key:id, value:list[id])
				ACPM sous forme de dictionnaire (value est une liste des sommet adjacent a la clé)
		"""

		n = len(self.distance_matrix) #nombre de sommets
		selected = [0] #ensemble des sommets visité 
		not_selected = [ i for i in range(1, n) ] #ensemble des sommets non visité 
		list_of_edges_selected = [] #liste des aretes séléctionnées
		T = {i: [] for i in range(n)}  #Dictionnaire -> (key:id, value:list des sommets adjacent dans T)


		while len(selected) < n: #Tant qu'on a pas séléctionné tous les somets
			min_cost = np.inf #on cherche l'arete de cout min reliant les sommets selectionnés à ceux non séléctionnés
			for i in selected:
				if verbose:
					print(f"{selected=}")
				for j in not_selected:
					cost_i_j = self.distance_matrix[i][j]
					if cost_i_j < min_cost: #plus petit cout vu actuellement
						min_cost = cost_i_j 
						s = j #sommet courant à ajouter dans l'ACPM
						f = i #pere courant de s
			if verbose:
				print(f"sommet ajouté: {s}")
				print(f"aretes ajoutée: {list_of_edges_selected=}")

			#Maj dic T (nouvelle aretes entre s et f)
			T[f].append(s)
			T[s].append(f)

			selected.append(s) #on ajoute le sommet s à l'arbre car il a le cout le plus petit avec l'ACPM courant
			list_of_edges_selected.append((f,s)) # on indique sont sommet adjacent pour savoir le relier a quel sommets dans l'ACPM
			not_selected.remove(s) # on enleve le sommet s des sommets non séléctionnés
		
		self.T = T
		self.list_of_edges_selected = list_of_edges_selected
		print("\nACPM :\n","T= ",T, "\nlist_of_edges_selected= ", list_of_edges_selected)
		return list_of_edges_selected, T
	
	def odd_degree_vertex(self):
		"""
			Renvoies les sommets de degré impaire de l'arbre T
			
		Args: 
			T: dict(key:int, value:list[int])
				dictionnaire du graphe value correspond au sommet adjacent a key

		Returns:
			odd_degree: list[int]
				liste des sommets de degré pairs dans T
		"""
		odd_degree = []
		for key in self.T:
			if len(self.T[key])%2 ==1: #Si sommet de degre impairs
				odd_degree.append(key)

		self.odd_degree = odd_degree
		print("\n ",len(self.odd_degree),"odd_degree_vertex find:\n", self.odd_degree)
		return odd_degree

	def inducted_distance_matrix(self):
		"""
			Renvoies la distance matrix induit par les sommets impaire de T		
		Args: 
		Returns:
		"""
		#créer la matrice de distance entre chaque points seulemnt des sommets de degré impaire
		inducted_distance_matrix = self.distance_matrix[np.ix_(self.odd_degree, self.odd_degree)] #matrix size = (len(self.odd_degree), len(self.odd_degree))
		self.inducted_distance_matrix = inducted_distance_matrix
		print("\ninducted distance matrix find:\nsize=",self.inducted_distance_matrix.shape,"\n", self.inducted_distance_matrix)
		return inducted_distance_matrix

	def minimum_weight_perfect_matching(self):
		G = nx.Graph()
		for u in self.odd_degree:
			for v in self.odd_degree:
				if u < v: #Pour éviter les répétition d'aretes
					w_uv = self.distance_matrix[u, v] #cost of edge (u,v)
					G.add_edge(u,v, weight=w_uv)
		
		matching = matching = nx.min_weight_matching(G)
		print("\n\nminimum_weight_perfect_matching:\n", matching)
		self.matching = list(matching)
		return self.matching
	
	def get_min_weight_perfect_matching(self):
		""" Utilise l'algorithme hongrois pour un couplage parfait de poids minimum """
		n = len(self.odd_degree)
		cost_matrix = np.full((n, n), np.inf)
		for i in range(n):
			for j in range(n):
				if i != j:
					u, v = self.odd_degree[i], self.odd_degree[j]
					cost_matrix[i][j] = self.distance_matrix[u][v]

		row_ind, col_ind = linear_sum_assignment(cost_matrix)

		matching = []
		used = set()
		for i, j in zip(row_ind, col_ind):
			if i < j and i not in used and j not in used:
				u, v = self.odd_degree[i], self.odd_degree[j]
				matching.append((u, v))
				used.add(i)
				used.add(j)

		self.matching = matching
		return matching
		
	def union_matching_ACPM(self):
		'''
		self.list_union_matching_ACPM = self.list_of_edges_selected + self.matching
		print("union_matching_ACPM:\n", self.list_union_matching_ACPM)
		'''
		""" Combine les arêtes de l’ACPM et du couplage pour former un multigraphe eulérien """
		self.merged_graph = defaultdict(list)
		for u, v in self.list_of_edges_selected + self.matching:
			self.merged_graph[u].append(v)
			self.merged_graph[v].append(u)

	def euler_tour(self):
		""" Trouve un tour eulérien avec l’algorithme de Hierholzer """
		graph = {u: list(vs) for u, vs in self.merged_graph.items()}
		circuit = []
		stack = []
		current = next(iter(graph))
		
		while stack or graph[current]:
			if not graph[current]:
				circuit.append(current)
				current = stack.pop()
			else:
				stack.append(current)
				next_node = graph[current].pop()
				graph[next_node].remove(current)
				current = next_node
				
		circuit.append(current)
		self.euler_path = circuit[::-1]
		return self.euler_path
	
	def euler_to_hamiltonian(self):
		""" Transforme le tour eulérien en une tournée hamiltonienne """
		visited = set()
		path = []
		for node in self.euler_path:
			if node not in visited:
				visited.add(node)
				path.append(node)
		path.append(path[0])  # retour au point de départ
		self.hamiltonian_path = path
		return path

	def launch(self):
		self.ACPM() #Pour le moment seulement ACPM de codé
		self.odd_degree_vertex()
		self.inducted_distance_matrix()
		# self.minimum_weight_perfect_matching()
		self.get_min_weight_perfect_matching()
		self.union_matching_ACPM()
		self.euler_tour()
		self.euler_to_hamiltonian()

		return self.hamiltonian_path
