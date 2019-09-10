import math

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

## 2D
'''
is the Euclidean distance given by
||x_i - x_j||=((x_i1-x_j1)^2 + (x_i2-x_j2)^2 + ... + (x_iD-x_jD)^2)^.5
'''
def euclidian_distance(p, q):
    return math.hypot(p[0]-q[0],p[1]-q[1])

'''
The Gaussian kernel is defined as
k(x_i,x_j)=exp(-|| x_i - x_j ||^2 / sigma^2)
'''
def gaussian_kernel(p, q, r):
    return math.exp(-euclidian_distance(p, q)**2/r**2)

def calc_gaussian(p, S, h):
    return sum([gaussian_kernel(p, q, h) for q in S])



def _euclidian_distance(p, q):
    return math.fabs(p-q)

def _gaussian_kernel(p, q, r):
    return math.exp(-_euclidian_distance(p, q)**2/r**2)





b=(5,5)
#S=[(1,1),(5,5),(99,99)]
S=[(4,1), (5,3), (7,1)]
h=1
#print (calc_gaussian(b, S, h))

o=0
for q in S:
	w = _gaussian_kernel(b[0], q[0], h)
	d = _euclidian_distance(b[0], q[0])
	o += w*d
	print (q[0],w,d, w*d)
print(o)

print ( (4+5+7)/3)



