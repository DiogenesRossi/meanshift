#https://pythonprogramming.net/mean-shift-from-scratch-python-machine-learning-tutorial/
#http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/AV0405/MISHRA/kde.html

from toy_clustering_datasets import *
from pylab import plot, grid, show


def mean_shift(S,r):
    P = []
    C = S
    while P != C:
        P = C
        C = sorted(set([centroid([q for q in C if distance(p,q)<r]) for p in C]))
    return C

def distance(p,q):
    return hypot(p[0]-q[0],p[1]-q[1])

def centroid(S):
    X = [p[0] for p in S]
    Y = [p[1] for p in S]
    return (sum(X)/len(X), sum(Y)/len(Y))


def t0():
    S = [(1,1), (1,2), (2,2),        # cluster 0
         (2,5), (3,4), (3,6), (4,5), # cluster 1
         (4,1), (5,3), (6,1)         # cluster 2
         ]

    L = [0, 0, 0,
         1, 1, 1, 1,
         2, 2, 2]
    
    show_clusters('Original clusters',S,L)
    C = mean_shift(S,1)
    title('Centroids found by Mean-Shift')
    for i,p in enumerate(S): scatter([p[0]],[p[1]],color=colour(L[i]),marker=marker(L[i]),s=25)
    for c in C: scatter([c[0]],[c[1]],color='k',marker='*',s=150)
    grid(True)
    show()


t0()
    

def t1():
    S = [
         (1.0,1.0), (1.0,2.0), (1.5,1.5), (2.0,1.0), (2.0,2.0), # cluster 0
        
         (3.5,1.5), (4.0,1.0), (4.0,2.0), (4.5,1.5), (4.5,2.5), # cluster 1
         (5.0,1.0), (5.0,2.0), (5.0,3.0), (5.5,1.5), (5.5,2.5),
         (6.0,2.0), (6.0,3.0), (6.5,2.5),
         
         (7.5,1.5), (8.0,1.0), (8.0,1.5), (8.0,2.0), (8.5,1.5), # cluster 2
         
         (9.0,9.0)                                              # noise
         ]

    L = [0, 0, 0, 0, 0,                                         # labels
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2,
         3]

    show_clusters('Original clusters',S,L)
    C = mean_shift(S,1)
    title('Centroids found by Mean-Shift')
    for i,p in enumerate(S): scatter([p[0]],[p[1]],color=colour(L[i]),marker=marker(L[i]),s=25)
    for c in C: scatter([c[0]],[c[1]],color='k',marker='*',s=150)
    grid(True)
    show()

#t1()
