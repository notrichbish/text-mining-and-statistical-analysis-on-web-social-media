import networkx as nx
import matplotlib.pyplot as plt
from cdlib import algorithms, viz 


#importing the data from Stanford Dataset Facebook Analysis
network_df = nx.read_edgelist("facebook_combined.txt.gz", create_using = nx.Graph(), nodetype=int)
print(nx.info(network_df))

network_df.degree(107)

posDf = nx.spring_layout(network_df)
nx.draw_networkx(network_df, pos = posDf,with_labels = False, node_size = 20)

#Degree Centrality
degree_cent = nx.degree_centrality(network_df)
node_color = [20000.0 * network_df.degree(v) for v in network_df]
node_size = [v * 10000 for v in degree_cent.values()]
plt.figure(figsize=(20,20))
nx.draw_networkx(network_df, pos = posDf, with_labels = False, node_color = node_color, node_size = node_size)
plt.axis('off')
sorted(degree_cent, key=degree_cent.get, reverse=True)[:5]

#Betweenness Centrality
betCent = nx.betweenness_centrality(network_df, normalized = True, endpoints = True)
node_color = [20000.0 * network_df.degree(v) for v in network_df]
node_size = [v * 10000 for v in betCent.values()]
plt.figure(figsize=(20,20))
nx.draw_networkx(network_df, pos = posDf, with_labels = False, node_color = node_color, node_size = node_size)
plt.axis('off')
sorted(betCent, key=betCent.get, reverse=True)[:5]


#Eigenvector Centrality
eigen_cent = nx.eigenvector_centrality(network_df)
node_color = [20000.0 * network_df.degree(v) for v in network_df]
node_size = [v * 10000 for v in eigen_cent.values()]
plt.figure(figsize=(20,20))
nx.draw_networkx(network_df, pos = posDf, with_labels = False, node_color = node_color, node_size = node_size)
plt.axis('off')
sorted(eigen_cent, key=eigen_cent.get, reverse=True)[:5]


#Community Detection and Network Clusters
coms = algorithms.louvain(network_df, weight = "weight", resolution = 1.)

viz.plot_network_clusters(network_df, coms, posDf)
viz.plot_community_graph(network_df,coms)
len(coms.communities)

