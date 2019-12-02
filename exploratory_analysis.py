import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
sns.set_context('talk')

# Plot Style
plt.style.use(u'ggplot')

# Read in training Data File
df = pd.read_json("../data/train.json")

# Most Common Ingredients
ingr_ind = Counter([ingredient for ingredient_list in df.ingredients for ingredient in ingredient_list])
ingr_ind = pd.DataFrame.from_dict(ingr_ind,orient='index').reset_index()
ingr_ind = ingr_ind.rename(columns={'index':'Ingredient', 0:'Count'})

# Plot Cuisine Distribution
f, ax = plt.subplots(figsize=(15,13))
sns.countplot(y = 'cuisine', data = df, order = df.cuisine.value_counts(ascending=False).index, palette = "RdBu")

# Plot Ingredient Distribution
f, ax = plt.subplots(figsize=(13,10))
sns.barplot(x = 'Count', y = 'Ingredient', data = ingr_ind.sort_values('Count', ascending=False).head(20), palette="RdBu")

# Least Common Ingredients
ingr_ind = Counter([ingredient for ingredient_list in train.ingredients for ingredient in ingredient_list])
ingr_ind = pd.DataFrame.from_dict(ingr_ind,orient='index').reset_index()
ingr_ind = ingr_ind.rename(columns={'index':'Ingredient', 0:'Count'})
ingr_ind.sort_values('Count', ascending=True).head(20)

# Plot Ingredient Count
f, ax = plt.subplots(figsize=(32,15))
sns.boxplot(x='cuisine',
            y='number_ingredients',
            data= (pd.concat([train.cuisine,train.ingredients.map(lambda l: len(l))], 
            axis=1).rename(columns={'ingredients':'number_ingredients'})), palette="Blues")

#print number of records
print("Number of records: {0}".format(len(df.id)))
print("---")
# printing cuisine types and counts
print("Percent of data from each cusine:")

print(df.cuisine.value_counts(normalize=True))
print("---")
print("Number of cuisine types: {0}".format(len(df.cuisine.value_counts())))
print("---")
print("Number of unique ingredients: {0}".format(len(set([ingredient for ingredient_list in df.ingredients.values for ingredient in ingredient_list]))))


