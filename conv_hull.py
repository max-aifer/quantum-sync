import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# create a set of N random points in 2D
N = 1000
points = np.random.randn(N,2)
plt.scatter(points[:,0], points[:,1])

# find the convex hull
hull = ConvexHull(points)

# plot the convex hull
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
# fill the convex hull
plt.fill(points[hull.vertices,0], points[hull.vertices,1], 'r', alpha=0.2)
plt.show()