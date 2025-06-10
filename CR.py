import numpy as np, random, matplotlib.pyplot as plt
from instances import *
from TSP import *
from collections import deque

class ConstructiveReasoning:
	def __init__(self, graph, hamiltonian_path):
		self.graph = graph
		self.hamiltonian_path = hamiltonian_path
		self.path = []
		self.visited = set()
		self.hamiltonian_path = hamiltonian_path
		self.current = hamiltonian_path[0]  # Le point de départ est toujours le premier point donné par Christofides
		self.path = [self.current]
		self.visited = {self.current}
		self.unvisited = hamiltonian_path[1:-1]  # Supprimer le premier et le dernier (le dernier est le même que le premier)
		self.direction = 1  # 1 = dans l'ordre normal, -1 = dans l'ordre inverse
		self.round_count = 1
		self.max_rounds = self.graph.n * 2  # Limite maximale pour éviter des boucles infinies
		print(" Chemin trouvé par l'algo de Christofides :", hamiltonian_path)
	
    
	# Renvoie les sommets visités entre src et dest selon la direction actuelle
	def get_subpath_nodes(self, src, dest, direction):
		idx_src = self.hamiltonian_path.index(src)
		idx_dest = self.hamiltonian_path.index(dest)
		n = len(self.hamiltonian_path)
		print("direction = ", direction)
		# Cas spécial : dest est directement après src
		if (idx_src + 1) % (n-1) == idx_dest:
			print(f"dest = {dest} d'indice {idx_dest=} est le prochain sommet après src = {src} d'indice {idx_src=} dans la liste hamiltonian_path")
			if direction == 1:
				return []

		if direction == 1:
			# Toujours avancer de src à dest en tournant si besoin 
			# on utilise le modulo pour faire le tour de la liste
			subpath = []
			i = (idx_src + 1) % n
			while i != idx_dest:
				if self.hamiltonian_path[i] in self.visited:
					subpath.append(self.hamiltonian_path[i])
				i = (i + 1) % n
			print(f"chemin sens normale = {subpath=}")
			return subpath

		else:  # direction == -1
			# Toujours reculer de src à dest en tournant si besoin
			subpath = []
			i = (idx_src - 1 + n) % n
			while i != idx_dest:
				if self.hamiltonian_path[i] in self.visited:
					subpath.append(self.hamiltonian_path[i])
				i = (i - 1 + n) % n
			print(f"chemin inversé = {subpath=}")
			return subpath


    # Tente de trouver un chemin de src à dest en utilisant uniquement les sommets déjà visités
	def try_shortcut(self, src, dest, direction):
		# print(f"argument try_shortcut : {src=}, {dest=} et {direction=}")
		
        # Cas src et dest sont directement connectés
		if self.graph.is_accessible(src, dest):
			return [src, dest]
		
        # la liste des nœuds visités entre src et dest dans le bon ordre
		allowed = self.get_subpath_nodes(src, dest, direction)
		print(f"allowed nodes : {allowed}")
		if not allowed:
			return None
		
        # Initialiser la recherche de chemin, ici avec BFS (Breadth-First Search)
		queue = deque([(src, [src])]) # File d'attente avec (sommet courant, chemin actuel)
		visited_inner = set() # Pour éviter de revisiter les mêmes sommets dans BFS
		
        # Parcours BFS pour essayer de relier src à dest
		while queue:
			print(f"queue : {queue=}")
			node, path_so_far = queue.popleft()
			
            # On verifie si on peux allez directement de node à dest
			if self.graph.is_accessible(node, dest): 
				return path_so_far + [dest]
			
			visited_inner.add(node)	
			
			# Sinon explorer tous les nœuds intermédiaires autorisés
			for neighbor in allowed + [dest]:
				if neighbor in visited_inner:
					continue
				# print(f"{self.graph.is_accessible(node, neighbor)=}, {node=}, {neighbor=}")
				if self.graph.is_accessible(node, neighbor):
					if neighbor == dest:
						return path_so_far + [neighbor]
					queue.append((neighbor, path_so_far + [neighbor]))
					visited_inner.add(neighbor)
		return None

    # Lance l'algorithme
	def run(self):
		last_endpoint = self.path[-1]  # Le point final de la ronde 0 est aussi le point de départ

        # Boucle principale : tant qu'il reste des sommets non visités
		while self.unvisited and self.round_count <= self.max_rounds:
			# Début de la ronde courante (fin de la ronde précédente)
			current_start = self.current
			ordered_unvisited = self._get_ordered_unvisited()
			
			if not ordered_unvisited:
				print("Aucun nœud à visiter cette ronde.")
				self.round_count += 1
				continue
			print(f"V_m (noeuds à visiter) : {ordered_unvisited}, nombre de noeuds : {len(ordered_unvisited)}")

			# Décider s’il faut changer de direction (basé sur les trois conditions de l'article)
			# Changement de direction sauf la premier rond
			if self.round_count > 1:
				# Condition : Le point de départ actuel ≠ point final précédent
				if current_start != last_endpoint:
					self.direction *= -1
					print("Changement de direction (v_m,0 ≠ v_{m-1,end})")

				# Condition : Le groupe de cette ronde = au groupe de la ronde suivante
				else:
					# Si on continue dans la même direction, regarder ce qu'on visiterait ensuite
					temp_direction = -self.direction
					next_unvisited = [v for v in self.hamiltonian_path if v not in self.visited]
					next_indices = [self.hamiltonian_path.index(v) for v in next_unvisited]
					next_indices.sort(reverse=(temp_direction == -1))
					ordered_next = [self.hamiltonian_path[i] for i in next_indices]

					if ordered_next == ordered_unvisited:
						print("Changement de direction (V_m == V_{m+1})")
						print(f"V_m     (direction actuelle): {ordered_unvisited}")
						print(f"V_m+1 (direction inversée): {ordered_next}")
						self.direction *= -1

			print(f"\nRonde {self.round_count} (sens = {'→' if self.direction == 1 else '←'})")

			round_path = [self.current] # Commencer avec le sommet actuel
			
            # Essayer de visiter tous les sommets prévus dans cette ronde
			for target in ordered_unvisited:
				self.graph.reveal_at(self.current)
				if target not in self.unvisited:
					continue
				shortcut = self.try_shortcut(self.current, target, self.direction)
				if shortcut:
					print(f"Raccourci trouvé de {self.current} à {target} via {shortcut}")
					for node in shortcut[1:]:
						if node not in self.visited:
							self.visited.add(node)
							self.unvisited.remove(node)
						self.path.append(node)
						round_path.append(node)
					self.current = target # Mettre à jour la position actuelle

            
			print(f"Fin de la ronde {self.round_count}, nœuds visités : {round_path}")
			print(f"Nœuds non visités : {list(self.unvisited)}")
			self.round_count += 1
			last_endpoint = self.current  # Mettre à jour le dernier point de la ronde
		
        # Protection anti-boucle infinie
		if self.round_count > self.max_rounds:
			print(f"Interruption : L'algorithme a dépassé le nombre maximal de {self.max_rounds} tours autorisés.")
			raise RuntimeError("Constructive Reasoning bloqué : dépassement du nombre de tours.")

		self._return_to_start()
		print("\nChemin final CR :", self.path)
		return self.path
	
    # Renvoie la liste des nœuds non visités ordonnés selon la direction actuelle.
	def _get_ordered_unvisited(self):
		indices = [self.hamiltonian_path.index(v) for v in self.unvisited]
		indices.sort(reverse=(self.direction == -1))
		return [self.hamiltonian_path[i] for i in indices]
	
    # Revient au point de départ à la fin du parcours
	def _return_to_start(self):
		start = self.hamiltonian_path[0]  # Revenir explicitement au point de départ (0)
		print(f"\nRetour vers le départ depuis {self.current}")
		
		shortcut = self.try_shortcut(self.current, start, self.direction)
		if not shortcut:
			self.direction *= -1
			print(f"Dernier changement de direction pour le retour")
			shortcut = self.try_shortcut(self.current, start, self.direction)
		
		# Si toujours impossible, lever une erreur
		if not shortcut:
			print(f"Impossible de retourner au départ depuis {self.current}")
			raise RuntimeError("Aucun chemin possible pour retourner au point de départ.")
		
		print(f"Chemin de retour via {shortcut}")
		for node in shortcut[1:]:
			self.path.append(node)