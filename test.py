import numpy as np
from math import pi
import matplotlib.pyplot as plt
from matplotlib import style
#style.use('gpp')
style.use('fivethirtyeight')

def generate_clusters(n_clusters, n_samples, noise_level=0.08, scale=8):
    """
    Returns N center points and (N,S) sample points that have a noise ratio of NL.
    
    The points will be scaled to with the given value
    """
    centroids = np.random.randn(n_clusters, 2)
    clusters = np.random.randn(n_clusters, n_samples, 2) * noise_level + np.expand_dims(centroids, axis=1)
    return centroids*scale, clusters*scale

def plot_clusters(centroids, clusters):
    """
    Print a chart with the centroids and clusters in different colors
    """
    colors = (list('bgrcmykw')*3)

    for i in range(len(centroids)):
        plt.scatter(clusters[i, :, 0], clusters[i, :, 1], c=colors[i])
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')


centroids, clusters = generate_clusters(5, 12)

plot_clusters(centroids, clusters)

#plt.show()

all_points = clusters.reshape(clusters.shape[0]*clusters.shape[1], -1)
print(all_points.shape)


def gaussian(x, std): 
    return np.exp(-0.5*((x/std)**2)) / (std * np.sqrt(2*pi))

def random_sample(size):
    """
    Returns size numbers between -5 and 5 in a np.array
    """
    return (np.random.sample(size) - 0.5) * 10

random_sample(10)

x = random_sample(100) 
plt.scatter(x, gaussian(x, 1))

#plt.show()


x = random_sample(200) 
x = x[x >= 0]

std = 1.7
plt.scatter(x, gaussian(x, std))
#plt.show()

point = all_points[0]
all_distances = np.sqrt((point - all_points) ** 2).sum(axis=1)
all_distances.shape

weights = gaussian(all_distances, 1.7)

new_x = (all_points * np.expand_dims(weights, 1)).sum(axis=0) / weights.sum()

print (new_x)

#http://www.clungu.com/Mean-Shift/#weights-kernel
