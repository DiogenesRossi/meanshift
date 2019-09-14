#https://pythonprogramming.net/mean-shift-from-scratch-python-machine-learning-tutorial/
#http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/AV0405/MISHRA/kde.html

from toy_clustering_datasets import *
from pylab import plot, grid, show
import math

# Opcoes para facilitar os testes
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-b", "--bandwidth", dest="bandwidth", help="define bandwidth: -b 2.1", metavar="BANDWIDTH", default="2.1")
parser.add_option("-k", "--kernel", dest="kernel",       help="choose kernel: -k [0:euclidian,1:gaussian]", metavar="KERNEL", default="1")
parser.add_option("-d", "--data", dest="data", help="choose data: [0,1]", metavar="DATA", default="0")
(opt, args) = parser.parse_args()
KERNEL = int(opt.kernel)
DATA = int(opt.data)
BANDWIDTH = float(opt.bandwidth)

#python mean_shift.py -d 0 -b 2.1 -k 0

def mean_shift(S,r):
    P = []
    C = S
    while P != C:
        P = C
        C = sorted(set([centroid(p, r, [q for q in C if distance(p,q)<r], KERNEL) for p in C]))
    return C

def distance(p,q):
    return hypot(p[0]-q[0],p[1]-q[1])

# https://mccormickml.com/2013/08/15/the-gaussian-kernel/
'''
Given two D dimensional vectors x_i and x_j. The Gaussian kernel is defined as
k(x_i,x_j)=exp(-|| x_i - x_j ||^2 / sigma^2)
where ||x_i - x_j|| is the Euclidean distance given by
||x_i - x_j||=((x_i1-x_j1)^2 + (x_i2-x_j2)^2 + ... + (x_iD-x_jD)^2)^.5
and sigma^2 is the bandwidth of the kernel.
Note that the Gaussian kernel is a measure of similarity between x_i and x_j.
It evalues to 1 if the x_i and x_j are identical, and approaches 0 as x_i and x_j move further apart.
The function relies on the dist function in the stats package for an initial estimate of the euclidean distance. 
'''
def gaussian_kernel(p, q, r):
    return math.exp(-distance(p, q)**2/r**2)

def calc_gaussian(p, S, h):
    o=0
    m=0
    for q in S:
        w = gaussian_kernel(p, q, h)
        o += w
        m += w*q[0]
    return m/o

def gaussian_centroid_by_axis(p, axis_data, h):
	o=0
	m=0
	for q in axis_data:
		w = math.exp(-math.fabs(p-q)**2/h**2)
		o += w
		m += w*q
	return m/o


def centroid(p, r, S, kernelId):
    m = 0
    X = [s[0] for s in S]
    Y = [s[1] for s in S]

    if (kernelId==0):
        m = (sum(X)/len(X), sum(Y)/len(Y)) # Calcula a media euclidiana para cada eixos
    else:
        m=(gaussian_centroid_by_axis(p[0], X, r), gaussian_centroid_by_axis(p[1], Y, r)) # calcula a distancia gaussiana para cada eixo
    return m
    
 
def get_data_sample(id):
    S=[]
    L=[]
    if id==0:
        S = [(1,1), (1,2), (2,2),        # cluster 0
             (2,5), (3,4), (3,6), (4,5), # cluster 1
             (4,1), (5,3), (6,1)         # cluster 2
            ]
        L = [0, 0, 0,
             1, 1, 1, 1, 
             2, 2, 2]

    elif id==1:
        S = [(1.0,1.0), (1.0,2.0), (1.5,1.5), (2.0,1.0), (2.0,2.0), # cluster 0
            (3.5,1.5), (4.0,1.0), (4.0,2.0), (4.5,1.5), (4.5,2.5),  # cluster 1
            (5.0,1.0), (5.0,2.0), (5.0,3.0), (5.5,1.5), (5.5,2.5),
            (6.0,2.0), (6.0,3.0), (6.5,2.5),
            (7.5,1.5), (8.0,1.0), (8.0,1.5), (8.0,2.0), (8.5,1.5),  # cluster 2
            (9.0,9.0)                                               # noise
            ]
        L = [0, 0, 0, 0, 0,                                         # labels
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2,
            3]

    return S,L


def main():
    S, L = get_data_sample(DATA)
    show_clusters('Original clusters',S,L)
    C = mean_shift(S, BANDWIDTH)
    title('Centroids found by Mean-Shift')
    for i,p in enumerate(S): scatter([p[0]],[p[1]],color=colour(L[i]),marker=marker(L[i]),s=25)
    for c in C: scatter([c[0]],[c[1]],color='k',marker='*',s=150)
    grid(True)
    show()

main()

