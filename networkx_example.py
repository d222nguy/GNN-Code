import networkx as nx
G = nx.Graph()
G.add_nodes_from([0, 1, 2, 3])
G.add_edges_from([(0, 1), (0, 2), (0, 3), (2, 3)])
a = nx.adjacen
nb_nodes = G.number_of_nodes()
nb_edges = G.number_of_edges()
print('Graph G with {0} nodes and {1} edges'.format(nb_nodes, nb_edges))
print('Adjacency list: {0}'.format(G.edges))
print('Adjacency matrix: ')
A = nx.adjacency_matrix(G)
print(A.todense())
print('Degree of node 1: {0}'.format(G.degree(1)))

#Draw
nx.draw(G, with_labels = True)