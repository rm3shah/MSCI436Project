# Import libraries
import pandas as pd
import numpy as np
import networkx as nx  
import json
import random
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# read in JSON into dataframe
def preprocess():
    dataset=[]
    with open('../data/train.json') as json_file:
        data = json.load(json_file)
        for recipe in data:
            dataset.append(recipe['ingredients'])
    return dataset

# plot support vs confidence graph 
def supp_vs_conf_graph(rules):
    support=rules['support'].values
    confidence=rules['confidence'].values
    for i in range (len(support)):
       support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5) 
       confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)

    plt.scatter(support, confidence, alpha=0.5, marker="*")
    plt.xlabel('support')
    plt.ylabel('confidence') 
    plt.show()

# define function to create association rule graph
def draw_graph(rules, num_rules):
    graph = nx.DiGraph()

    color_map=[]
    N = 50
    colors = np.random.rand(N)       

    for i in range(num_rules): 
        graph.add_node("R"+str(i))

        for a in rules.iloc[i]['antecedents']:
            graph.add_node(a)
            graph.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)

        for c in rules.iloc[i]['consequents']:
            graph.add_node(c)
            graph.add_edge("R"+str(i), c, color=colors[i],  weight=2)

    edges = graph.edges()
    colors = [graph[u][v]['color'] for u,v in edges]
    weights = [graph[u][v]['weight'] for u,v in edges]

    pos = nx.spring_layout(graph, k=16, scale=1)
    nx.draw(graph, pos, edges=edges, node_color = color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)            

    for p in pos:  # raise text positions
        pos[p][1] += 0.07
    nx.draw_networkx_labels(graph, pos)
    plt.show()

def create_rules(dataset):
    # encode data for apriori algo
    encoder = TransactionEncoder()
    fit_data = encoder.fit(dataset).transform(dataset)
    encoded_df = pd.DataFrame(fit_data, columns=encoder.columns_)
    
    # define frequent itemsets
    frequent_itemsets = apriori(encoded_df, min_support=0.05, use_colnames=True).sort_values('support', ascending=False)
    
    # create association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2).sort_values('confidence', ascending=False)
    # print(rules[['antecedents', 'consequents', 'support', 'confidence']])
    
    # draw association graph
    draw_graph(rules, 10)
    
    # draw support vs confidence graph
    supp_vs_conf_graph(rules)


# call all functions 
dataset = preprocess()
create_rules(dataset)