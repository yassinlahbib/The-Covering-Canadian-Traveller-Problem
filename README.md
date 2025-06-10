# Problème du voyageur de commerce avec arêtes bloquées

Dans ce projet, nous nous intéressons au **problème du voyageur de commerce** (TSP) avec contraintes de blocages sur les arêtes du graphe. L'objectif est de construire des tournées de coût minimal, tout en s'adaptant dynamiquement aux blocages rencontrés lors du parcours.

## Objectifs du projet

- Développer et comparer deux algorithmes principaux :
  - **Cycling Routing (CR)**
  - **Cyclic Nearest Neighbor (CNN)**
- Étudier les performances des solutions générées à travers des analyses graphiques.

---

## Organisation des fichiers

- `instance.py`  
  Contient la configuration principale du projet, y compris :
  - l'importation du graphe ;
  - l'implémentation des méthodes communes à tous les algorithmes ;
  - la configuration des paramètres d'exécution.

- `instances/`  
  Contient les différentes instances prédéfinies de graphe utilisées pour les tests.

- `CCTPGraph.py`  
  Contient deux classes principales :
  - `CCTPGraph` : création et gestion des graphes avec arêtes bloquées ;
  - `Christophides` : implémentation de l'algorithme de Christofides pour générer des solutions initiales au TSP.

- `CNN.py`  
  Contient le code de l'algorithme CNN, conçu pour gérer dynamiquement les blocages.

- `CR.py`  
  Contient le code de l'algorithme Constructive Reasoning (CR).

- `etudes.py`  
  Contient les outils d'analyse et de visualisation graphique des performances des algorithmes.

- `figures/`  
  Contient les figures générées par `etudes.py` lors des analyses expérimentales.