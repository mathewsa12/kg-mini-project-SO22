from rdflib import Graph,URIRef
import pandas as pd
import numpy as np
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.models import TransE
import torch
from sklearn import preprocessing
from rdflib.namespace import FOAF, NamespaceManager
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score

import warnings
warnings.filterwarnings("ignore")

def convertToTriples(graph):
    return np.asarray([[s,p,o] for s,p,o in graph])

def main():
	le = preprocessing.LabelEncoder()

	cgns = Graph()
	cgns.parse("carcinogenesis/carcinogenesis.owl")

	triples = convertToTriples(cgns)
	df = pd.DataFrame(data = triples,columns = ['s','p','o'])

	tf_owl = TriplesFactory.from_labeled_triples(triples=df.to_numpy())

	owl_triples = TransE(triples_factory = tf_owl)

	owlhrtscores = owl_triples.score_hrt(tf_owl.mapped_triples).detach().numpy()
	df['hrt'] = owlhrtscores

	g = Graph()
	g.parse("kg22-carcinogenesis_lps1-train.ttl")

	trainTriples = convertToTriples(g)
	train_df = pd.DataFrame(data = trainTriples,columns = ['lp','label','s'])

	g_test = Graph()
	g_test.parse("kg22-carcinogenesis_lps2-test.ttl")

	test_triples = convertToTriples(g_test)
	test_df = pd.DataFrame(data = test_triples,columns = ['lp','label','s'])

	tf_test = TriplesFactory.from_labeled_triples(triples=test_df.to_numpy())

	test_triples = TransE(triples_factory = tf_test)

	testhrtscores = test_triples.score_hrt(tf_test.mapped_triples).detach().numpy()
	test_df['hrt'] = testhrtscores

	all_df = pd.merge(df,test_df,left_on='s',right_on='s',how='inner')
	le = preprocessing.LabelEncoder()
	all_df.drop('hrt_x',axis = 1,inplace = True)
	all_df['labelencodeds'] = le.fit_transform(all_df['s'])

	labels = all_df['lp'].unique()
	remaining_data = pd.DataFrame(data = [])
	total = 0
	for lp in labels:
	    label_df = all_df[all_df['lp'] == lp]
	    remaining_data_in_oneLP = df[~df['s'].isin(label_df['s'])].copy()
	    remaining_data_in_oneLP['lp'] = lp
	    remaining_data = remaining_data.append(remaining_data_in_oneLP,ignore_index = True)
	    remaining_data.drop_duplicates()
	remaining_data["labelencodeds"] = le.fit_transform(remaining_data["s"])


	X = all_df[["labelencodeds","hrt_y"]]
	y = all_df["label"]
	X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.33, random_state=42)

	classifier= KNeighborsClassifier(metric='minkowski', p=2 )  
	classifier.fit(X_train, y_train)

	y_pred = classifier.predict(X_test)
	# print("Macro F1-Score: ", f1_score(y_test, y_pred, average="macro"))
	# print("Accuracy score:", accuracy_score(y_test,y_pred))

	Final_test = remaining_data[["labelencodeds","hrt"]]
	final_prediction = classifier.predict(Final_test)
	remaining_data["label"] = final_prediction
	remaining_data.head

	final_dataframe = remaining_data[['s','label','lp','p']]
	final_dataframe = final_dataframe[final_dataframe['label'] == "https://lpbenchgen.org/property/includesResource"]

	final_graph = Graph()
	final_graph.namespace_manager.bind('lpclass', URIRef('https://lpbenchgen.org/class/'))
	final_graph.namespace_manager.bind('carcinogenesis', URIRef('http://dl-learner.org/carcinogenesis#'))
	final_graph.namespace_manager.bind('rdf', URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#'))
	final_graph.namespace_manager.bind('lpres', URIRef('https://lpbenchgen.org/resource/'))
	final_graph.namespace_manager.bind('lpprop', URIRef('https://lpbenchgen.org/property/'))
	for i, row in final_dataframe.iterrows():
		final_graph.add((URIRef(row['lp']), URIRef(row['label']), URIRef(row['s'])))

	for i in labels: 
		final_graph.add((URIRef(i), URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'), URIRef('https://lpbenchgen.org/class/LearningProblem')))

	final_graph.serialize(destination = 'classification_result.ttl', format='ttl')

	pass


if __name__ == '__main__':
	main()
    