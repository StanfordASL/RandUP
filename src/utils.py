import numpy as np
import math
from scipy.spatial import ConvexHull, Delaunay
from scipy.special import betainc, beta, gamma

def is_in_convex_hull(p, hull):
    """
    Reference:
    https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
    
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0
def is_hull1_a_subset_of_hull2(hull_1, hull_2):
    pts_1 = hull_1.points[hull_1.vertices,:]
    hull_2_delaunay = Delaunay(hull_2.points)
    for pt in pts_1:
        if not(is_in_convex_hull(pt, hull_2_delaunay)):
            return False
    return True
def distance_point_to_segment(point, seg_pt_1, seg_pt_2):
    # https://stackoverflow.com/questions/41000123/computing-the-distance-to-a-convex-hull
    x1,y1 = seg_pt_1
    x2,y2 = seg_pt_2
    x3,y3 = point
    px = x2-x1
    py = y2-y1
    something = px*px + py*py + 1e-6
    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(something)
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    x = x1 + u * px
    y = y1 + u * py
    dx = x - x3
    dy = y - y3
    dist = np.sqrt(dx*dx + dy*dy)
    return dist
def distance_point_to_convex_hull(point, hull):
    pts_hull = hull.points[hull.vertices,:2]
    pts_hull = np.append(pts_hull, pts_hull[0,:][np.newaxis,:], axis=0)
    dists = np.zeros(pts_hull.shape[0])
    for i in range(pts_hull.shape[0]):
        if i<pts_hull.shape[0]-1:
            pt_hull_1 = pts_hull[i,:]
            pt_hull_2 = pts_hull[i+1,:]
        else:
            pt_hull_1 = pts_hull[-1,:]
            pt_hull_2 = pts_hull[0,:]
        dists[i] = distance_point_to_segment(point, pt_hull_1, pt_hull_2)
    return np.min(dists)
def Hausdorff_dist_two_convex_hulls(hull_1, hull_2):
    pts_hull_1 = hull_1.points[hull_1.vertices,:2]
    pts_hull_2 = hull_2.points[hull_2.vertices,:2]
    dists12 = np.zeros(pts_hull_1.shape[0])
    dists21 = np.zeros(pts_hull_2.shape[0])
    for i, pt in enumerate(pts_hull_1):
        dists12[i] = distance_point_to_convex_hull(pt, hull_2)
    for i, pt in enumerate(pts_hull_2):
        dists21[i] = distance_point_to_convex_hull(pt, hull_1)
    return np.maximum(np.max(dists12), np.max(dists21))
def are_points_in_ball(center, radius, points):
    # Inputs:
    #   center - (xdim,)
    #   radius - scalar
    #   points - (M,xdim)
    # Outputs:
    #   B_are_in_ball - (M,) (vector of booleans)
    return np.linalg.norm(points-center[None,:], axis=1)<=radius
def dist_points_to_ball(center, radius, points):
    # Inputs:
    #   center - (xdim,)
    #   radius - float
    #   points - (M,xdim)
    # Outputs:
    #   dists_to_ball - (M,) (vector of floats)
    return np.maximum(0, np.linalg.norm(points-center[None,:], axis=1)-radius)
def dist_points_to_sphere(center, radius, points):
    # Inputs:
    #   center - (xdim,)
    #   radius - float
    #   points - (M,xdim)
    # Outputs:
    #   dists_to_sphere - (M,) (vector of floats)
    return np.abs(np.linalg.norm(points-center[None,:], axis=1)-radius)
def Hausdorff_dist_ball_hull(ball_c, ball_r, hull):
    # Inputs:
    #   center - (xdim,)
    #   radius - float
    #   hull   - scipy.spatial.ConvexHull
    #   points - (M,xdim)
    # Outputs:
    #   haus_dist - float
    if ball_c.shape[0]>2:
        raise NotImplementedError("Not implemented for n_x > 2.")
    angs = np.arange(0, 2*np.pi, 0.01)
    sampled_pts = np.concatenate([ball_c[0,None,None]+ball_r*np.cos(angs)[:,None],
                                  ball_c[1,None,None]+ball_r*np.sin(angs)[:,None]], axis=1)
    hull_ball = ConvexHull(sampled_pts)
    return Hausdorff_dist_two_convex_hulls(hull, hull_ball)

def volume_intersection_ndim_balls(n, d, r1, r2):
    # n: dimension of the two balls
    # d: distance between the centers of the two balls
    # r1, r2: radius of each ball
    # https://math.stackexchange.com/questions/162250/how-to-compute-the-volume-of-intersection-between-two-hyperspheres
    # http://docsdrive.com/pdfs/ansinet/ajms/2011/66-70.pdf
    c1 = (d**2+r1**2-r2**2)/(2*d)
    c2 = (d**2-r1**2+r2**2)/(2*d)
    
    I_1 = betainc((n+1)/2.0, 0.5, 1.0-(c1**2)/(r1**2))
    I_2 = betainc((n+1)/2.0, 0.5, 1.0-(c2**2)/(r2**2))
    I_1 = I_1 * beta((n+1)/2.0, 0.5)
    I_2 = I_2 * beta((n+1)/2.0, 0.5)
    
    vol_cap_1 = 0.5 * (np.pi**(n/2)/gamma(n/2+1)) * (r1**n) * I_1
    vol_cap_2 = 0.5 * (np.pi**(n/2)/gamma(n/2+1)) * (r2**n) * I_2
    
    total_volume = vol_cap_1 + vol_cap_2
    return total_volume