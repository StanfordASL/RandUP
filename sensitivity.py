import numpy as np
import scipy.spatial
from scipy.optimize import fsolve, bisect
from scipy.stats import beta as beta_distribution
import matplotlib.pyplot as plt

import matplotlib.pylab as pl
from src.stats import sample_pts_ellipsoid_surface
from src.stats import volume_ellipsoid
from src.utils import Hausdorff_dist_two_convex_hulls
from src.utils import volume_intersection_ndim_balls
from src.viz import plot_ellipse

np.random.seed(0)

# ----------------------
# Number of samples
M = 1000
# ----------------------

# ----------------------
# Ball input set, with ellipsoidal parameterization
#     (xi-X_mu)^T X_Q^{-1} (xi-X_mu) <= 1.
x_dim = 2
r    = 1. # not tested with r not equal to 1.
X_mu = np.zeros(x_dim)
X_Q  = r**2*np.eye(x_dim)
# # Reachability map
def f(x, L):
	# x - (2, M) with M number of samples
	# L - scalar
	A = np.array([[L,0.],[0.,1.]])
	return A @ x
def Y_Q_matrix(X_Q, L):
	# X_Q - (2, 2) - Q-shape matrix parameterizing ellpisoidal set Y,
	#                such that (yi-Y_mu)^T Y_Q^{-1} (yi-Y_mu) <= 1.
	# L - scalar
	A = np.array([[L,0.],[0.,1.]])
	Y_Q = A @ X_Q @ A.T
	return Y_Q
# ----------------------

# ----------------------
def weighted_sample_pts_unit_ball(dim, NB_pts, alpha=1.0, beta=1.0):
    """
    Uniformly samples points on a d-dimensional sphere (boundary of a ball)
    Points characterized by    ||x||_2 = 1
    arguments:  dim    - nb of dimensions
                NB_pts - nb of points
                random - True: Uniform sampling. 
                         False: Uniform deterministic grid 
    output:     pts    - points on the boundary of the sphere [xdim x NB_pts]
    Reference: http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    """
    us    = np.random.normal(0,1,(dim,NB_pts))
    norms = np.linalg.norm(us, 2, axis=0)
    if alpha==1. and beta==1.:
    	# uniform distribution
    	rs = np.random.random(NB_pts)**(1.0/dim)
    else:
    	rs = np.random.beta(alpha, beta, size=(NB_pts))**(1.0/dim)
    pts   = rs*us / norms
    return pts
def weighted_sample_pts_in_ellipsoid(mu, Q, NB_pts, alpha=1.0, beta=1.0):
    """
    Uniformly samples points in an ellipsoid, specified as
            (xi-mu)^T Q^{-1} (xi-mu) <= 1
    arguments: mu - mean [dim]
                Q - Q [dim x dim]
    output:     pts - points sampled uniformly in ellipsoid [xdim x NB_pts]
    """
    xs = weighted_sample_pts_unit_ball(mu.shape[0], NB_pts, alpha, beta)
    E  = np.linalg.cholesky(Q)
    ys = (np.array(E@xs).T + mu).T
    return ys
# ----------------------

# ----------------------
def D_max_packing_nb_ball(dim, r, eps):
	D = 0.
	if dim == 2:
		D = (2.*np.pi*r) / (2.*eps) + 1   # place balls on circle
	else:
		D = (2*(2*r)/eps + 1.)**dim # Dumbgen, Walther, Rates of (...), 1996
	return D
def bound_conservatism_prob(dim, r, L, eps, M, alpha, beta):
	Q = r**2*np.eye(dim)
	eps_bar = eps/(2*L)
	D = D_max_packing_nb_ball(dim, r, eps_bar)
	vol_sampling = volume_intersection_ndim_balls(dim, r, r, eps_bar)
	p0lambda_2 = vol_sampling/volume_ellipsoid(Q)

	# Assumes r=1
	distrib  = beta_distribution(alpha, beta)
	p0alpha  = (1. - distrib.cdf((r-eps_bar)**dim))
	p0alpha /= (1-(r-eps_bar)**dim)
	p0lambda_2 = p0alpha * p0lambda_2

	delta_M = D*((1-p0lambda_2)**M)
	return np.maximum(1-delta_M, 0.)
# ----------------------

# ----------------------
beta = 1.

N_alphas = 10
N_lipsch = 10

alpha_max = 100
L_max     = 3

alpha_vec = np.geomspace(1., alpha_max, num=N_alphas)
L_vec     = np.linspace(1., L_max, num=N_lipsch)

N_runs = 100

haus_dists_exp  = np.zeros((N_alphas,N_lipsch,N_runs))
haus_dists_theo = np.zeros((N_alphas,N_lipsch))
for j, L in enumerate(L_vec):
	print("L =", L)
	Y_mu        = f(X_mu, L)
	Y_Q         = Y_Q_matrix(X_Q, L)
	ys_true     = sample_pts_ellipsoid_surface(Y_mu, Y_Q, NB_pts=1000)
	Y_true_hull = scipy.spatial.ConvexHull(ys_true.T)
	for i, alpha in enumerate(alpha_vec):
		# Compute the theoretical bound
		delta = 0.001
		f_bound_epsilon = lambda epsilon: bound_conservatism_prob(x_dim, r, L, epsilon, M, alpha, beta)
		f_delta_eps = lambda epsilon: f_bound_epsilon(epsilon)-delta
		eps_sol = bisect(f_delta_eps, 1e-7, r)
		haus_dists_theo[i,j] = eps_sol

		# Sample to get an empirical estimate
		for k in range(N_runs):
			xs = weighted_sample_pts_in_ellipsoid(X_mu, X_Q, M, alpha, beta)
			ys = f(xs, L)
			Y_est_hull = scipy.spatial.ConvexHull(ys.T)
			haus_dists_exp[i,j,k] = Hausdorff_dist_two_convex_hulls(Y_true_hull, Y_est_hull)
# ----------------------

# ----------------------
# Plot
colors = pl.cm.bwr(np.linspace(0,1,N_lipsch))
Z = [[0,0],[0,0]]
CS3 = plt.contourf(Z, np.linspace(1.,L_max,num=201), cmap=pl.cm.bwr)
plt.clf()
for j in range(N_lipsch):
	haus = haus_dists_exp[:,j,:]
	mean_haus, stds_haus = np.mean(haus, 1), np.sqrt(np.var(haus, axis=1))
	plt.plot(alpha_vec, mean_haus, 
			 color=colors[j], linewidth=2)
	# plt.fill_between(alpha_vec, mean_haus-3.*stds_haus, mean_haus+3.*stds_haus, 
 #                                     color=colors[j], alpha=0.2)
	plt.plot(alpha_vec, haus_dists_theo[:,j], 
			 color=colors[j], linestyle="--", linewidth=2)

plt.xlabel(r'$\alpha$', fontsize=20)
plt.ylabel(r'$d_H(\hat{Y}^M,Y)$', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().set_yscale('log')
cbar = plt.colorbar(CS3, ticks=[1, 2, L_max])
cbar.set_label(r'$L$', fontsize=20, rotation='horizontal', labelpad=16, y=0.53)
cbar.ax.tick_params(labelsize=16) 
plt.grid(which='minor', alpha=0.5, linestyle='--')
plt.grid(which='major', alpha=0.75, linestyle=':')
plt.ylim([1.9e-3,0.55])
plt.show()
# ----------------------