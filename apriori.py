# Import necessary python libraries
import pandas as pd
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# read in JSON into array of arrays
dataset=[]
with open('train.json') as json_file:
    data = json.load(json_file)
    for recipe in data:
        dataset.append(recipe['ingredients'])
for recipe in dataset:
    print(recipe)


# Transform your data for the apriori algorithm
encoder = TransactionEncoder()
arr = encoder.fit(dataset).transform(dataset)
df = pd.DataFrame(arr, columns=encoder.columns_)
df

frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)
frequent_itemsets

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
print(rules[['antecedents', 'consequents', 'support', 'confidence']])

support=rules['support'].values
confidence=rules['confidence'].values
for i in range (len(support)):
   support[i] = support[i] + 0.0025 * (random.randint(1,10) - 5) 
   confidence[i] = confidence[i] + 0.0025 * (random.randint(1,10) - 5)
 
plt.scatter(support, confidence, alpha=0.5, marker="*")
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()

def draw_graph(rules, rules_to_show):
  G1 = nx.DiGraph()
   
  color_map=[]
  N = 50
  colors = np.random.rand(N)       
   
  for i in range(rules_to_show): 
    G1.add_node("R"+str(i))
    
    for a in rules.iloc[i]['antecedents']:
        G1.add_node(a)
        
        G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)
       
    for c in rules.iloc[i]['consequents']:
             
            G1.add_node(c)
            
            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)
   
  edges = G1.edges()
  colors = [G1[u][v]['color'] for u,v in edges]
  weights = [G1[u][v]['weight'] for u,v in edges]
 
  pos = nx.spring_layout(G1, k=16, scale=1)
  nx.draw(G1, pos, edges=edges, node_color = color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
  for p in pos:  # raise text positions
           pos[p][1] += 0.07
  nx.draw_networkx_labels(G1, pos)
  plt.show()

  draw_graph(rules, 6)