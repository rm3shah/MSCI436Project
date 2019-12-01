# Import libraries
import pandas as pd
import numpy as np
import networkx as nx  
import json
import random
import collections
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# read in JSON into dataframe
def preprocess():
    with open('../data/train.json') as json_file:
        df = pd.read_json(json_file, orient='columns')
        cuisines = df.cuisine.unique()
    return df, cuisines

# get all the recipes for a cuisine
def get_cuisine_data(df, cuisine):
    data = df[df['cuisine']==cuisine].ingredients
    return data

# create a dictionary mapping cuisine to all recipes of that cuisine type
def create_cuisine_dict(df, cuisines):
    cuisine_dict = {}
    for cuisine in cuisines:
        cuisine_dict["{0}".format(cuisine)] = get_cuisine_data(df, cuisine)
    return cuisine_dict

# remove the 12 most common ingredients
def remove_common_ingredients(cuisine_dict):
    common_ingredients = ['salt','olive oil', 'onions', 'water', 'garlic', 'sugar', 'garlic cloves', 'butter', 'ground black pepper', 'all-purpose flour', 'pepper', 'vegetable oil']
    for cuisine in dataset[1]:
        for recipe in cuisine_dict[cuisine]:
            for word in list(recipe):
                if word in common_ingredients:
                    recipe.remove(word)
    return cuisine_dict

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

def create_rules(cuisine_data):
    # encode data for apriori algo
    encoder = TransactionEncoder()
    fit_data = encoder.fit(cuisine_data).transform(cuisine_data)
    encoded_df = pd.DataFrame(fit_data, columns=encoder.columns_)
    
    # define frequent itemsets
    frequent_itemsets = apriori(encoded_df, min_support=0.05, use_colnames=True).sort_values('support', ascending=False)
    
    # create association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
    # print(rules[['antecedents', 'consequents', 'support', 'confidence']])
    
    # draw graph
    draw_graph(rules, 10)


# call all functions 
dataset = preprocess()
cuisine_dict = create_cuisine_dict(dataset[0], dataset[1])
filtered = remove_common_ingredients(cuisine_dict)
for cuisine in dataset[1]:
    create_rules(filtered[cuisine])
    print(cuisine)
