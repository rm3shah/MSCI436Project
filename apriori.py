# Import necessary python libraries
import pandas as pd
import csv
#from mlxtend.preprocessing import TransactionEncoder
#from mlxtend.frequent_patterns import apriori, association_rules


#Maybe do this for json instead of csv
dataset = pd.read_json("../input/train.json")
dataset['ingredients_clean_string'] = [' , '.join(z).strip() for z in dataset['ingredients']]  
dataset['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in dataset['ingredients']]       

# Read in CSV file into an array of arrays
#dataset = []
#with open('aprioriData1.csv') as f:
#	reader = csv.reader(f)
#	for row in reader:
#		dataset.append(row)
#for row in dataset: 
#	print(row)

# Transform your data for the apriori algorithm
oht = TransactionEncoder()
oht_ary = oht.fit(dataset).transform(dataset)
df = pd.DataFrame(oht_ary, columns=oht.columns_)
print(df)

frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print(rules[['antecedents', 'consequents', 'support', 'confidence']])


