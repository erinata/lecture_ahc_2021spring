import pandas

from sklearn.preprocessing import normalize

import matplotlib.pyplot as pyplot

import scipy.cluster.hierarchy as shc

from sklearn.cluster import AgglomerativeClustering


dataset = pandas.read_csv("dataset_more_groups.csv")

print(dataset)

dataset_normal = pandas.DataFrame(normalize(dataset))

print(dataset_normal)


pyplot.title("Dendrogram")
dendrogram_object = shc.dendrogram(shc.linkage(dataset_normal, method='ward'))
pyplot.savefig("den.png")
pyplot.close()


machine = AgglomerativeClustering(n_clusters=6, affinity="euclidean", linkage="ward")
results = machine.fit_predict(dataset_normal)

pyplot.scatter(dataset['x1'], dataset['x2'], c=results )
pyplot.savefig("scatterplot_color.png")
pyplot.close()

print(results)













