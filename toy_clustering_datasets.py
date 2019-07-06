from math import *
from random import *
from pylab import plot, scatter, grid, title, xlabel, ylabel, show
from matplotlib import colors

UNKNOWN = -1
NOISE   = -2

def colour(n):
    C = ['blue','red','green','yellow','magenta','cyan'] + list(colors.cnames.keys())
    return C[n%len(C)]


def marker(n):
    M = ['o','s','D','p','h','d','H','v','^','<','>','8','*','+','1','2','3','4']
    return M[n%len(M)]

def show_clusters(T,S,L):
    '''
       Show the scatter plot for a clustering
       
       Parameters
       ----------
       T : title of the plot
       S : set of data points (a list of tuple of numbers)
       L : list of labels for points of S (labels are integer numbers,
           -1 means UNKNOWN and -2 means NOISE data)
    '''
    C = ['blue','red','green','yellow','magenta','cyan'] + list(colors.cnames.keys())
    M = ['o','s','D','p','h','d','H','v','^','<','>','8','*','+','1','2','3','4']
    for i in sorted(set(L)):
        (X,Y) = coordinates(S,L,i)
        scatter(X,Y,s=50,c=C[i%len(C)],marker=M[i%len(M)])
    (X,Y) = coordinates(S,L,NOISE)
    scatter(X,Y,s=50,c='k',marker='x')
    grid(True)
    title(T)
    xlabel('x')
    ylabel('y')
    show()


def coordinates(S,L,k):
    '''
       Split coordinates of points of S that belong to cluster with label k
       
       Parameters
       ----------
       S : set of data points (a list of tuple of numbers)
       L : list of labels for points of S (labels are integer numbers,
           -1 means UNKNOWN and -2 means NOISE data)
       k : label of the cluster
       
       Returns
       -------
       A tuple (X,Y), where X and Y are the lists of coordinates of the points in cluster k
    '''
    P = [p for i,p in enumerate(S) if L[i]==k]
    X = [x for (x,y) in P]
    Y = [y for (x,y) in P]
    return X,Y


def make_blobs(B):
    '''
       Create Gaussian blobs
       
       Parameters
       ----------
       B : list of tuples of the form (x,y,d,n), where
           (x,y) is the center, d is the standard deviation,
           and n is the number of points in the blob
       
       Returns
       -------
       S : A set of random data points, with normal distribution (a list of tuples)
       L : Labels for clustering membership of each point (list of integer numbers)
    '''
    C = []
    for i in range(len(B)): C.extend([(i,p) for p in blob(*B[i])])
    shuffle(C)
    return [p for i,p in C], [i for i,p in C]
    

def blob(x, y, d, n):
    '''
       Create a Gaussian blob
       
       Parameters
       ----------
       x : x coordinate of blob's center (a float number)
       y : y coordinate of blob's center (a float number)
       d : standard deviation of the blob
       n : number of points in the blob
       
       Returns
       -------
       A set of data points (a list of tuples)
    '''
    return [(gauss(x,d),gauss(y,d)) for i in range(n)]

##def make_blobs(n_points, blobs):
##    n = round(n_points/len(blobs))
##    m = n_points - n*(len(blobs)-1)
##    S = []
##    for i,(xc,yc,noise) in enumerate(blobs):
##        B = transf([(0,0)]*(n if i+1<len(blobs) else m),xc,yc,1,noise)
##        S.extend((b,i) for b in B)
##    shuffle(S)
##    return [p for (p,_) in S], [i for (_,i) in S]

def transf(S, x_center, y_center, radius, noise):
    return [(x_center + radius*x + gauss(0,noise),
             y_center + radius*y + gauss(0,noise))
            for (x,y) in S]

def make_circles(n_points, noise):
    n1 = 3*n_points//8
    n2 = n_points - n1
    C1 = transf(circle(n1), 0.0, 0.0, 0.3, noise)
    C2 = transf(circle(n2), 0.0, 0.0, 1.0, noise)
    return C1+C2, n1*[0]+n2*[1]

def circle(n_points):
    return [(cos(i*2*pi/n_points),sin(i*2*pi/n_points)) for i in range(n_points)]

def make_moons(n_points, noise):
    n1 = n_points//2
    n2 = n_points - n1
    M1 = transf(semicircle(n1, +1), 0.0, 0.0, 1, noise)
    M2 = transf(semicircle(n1, -1), 1.0, 0.5, 1, noise)
    return M1+M2, n1*[0]+n2*[1]

def semicircle(n_points, sign):
    return [(cos(i*pi/(n_points-1)),sign*sin(i*pi/(n_points-1))) for i in range(n_points)]




