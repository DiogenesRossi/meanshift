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
    return math.hypot(p[0]-q[0], p[1]-q[1])

'''e
The Gaussian kernel is defined as
k(x_i,x_j)=exp(-|| x_i - x_j ||^2 / sigma^2)
'''
def gaussian_kernel(p, q, r):
    return math.exp(-euclidian_distance(p, q)**2/r**2)

b=(7,7)
S=[(1,1), (2,2), (3,3), (9,9), (10,10), (11,11), (12,12)]
h=1
print("Data: ", S)
print("point: ", b)

def gaussian_centroid():
    o=0
    m=0
    for q in S:
        w = gaussian_kernel(b, q, h)
        o += w
        m += w*q[0]

    return(m/o)

def gaussian_centroidByAxis(p, axis_data):
    o=0
    m=0
    for q in axis_data:
        w = math.exp(-math.fabs(p-q)**2/h**2)
        o += w
        m += w*q
    return m/o

def main():
    X = [s[0] for s in S]
    Y = [s[1] for s in S]

    print (gaussian_centroidByAxis(b[0], X))
    print (gaussian_centroidByAxis(b[1], Y))

    print(gaussian_centroid())

main()



