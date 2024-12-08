import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import spdiags
from scipy.integrate import solve_ivp
from scipy.fftpack import ifft2, fft2
from scipy.sparse.linalg import bicgstab, gmres
from scipy.linalg import lu, solve, solve_triangular

# Method using Fourier Transform
def rhs_fourier(t, omega0):
    omega = omega0.reshape((n, n))
    omegat = fft2(omega) 
    psit = -omegat/k_squared
    psi = np.real(ifft2(psit))
    psi = psi.reshape(m**2)
    dpsi_dx = B @ psi
    dpsi_dy = C @ psi
    domega_dx = B @ omega0
    domega_dy = C @ omega0
    J = dpsi_dx * domega_dy - dpsi_dy * domega_dx  # Element-wise product

    # Compute Laplacian of omega (diffusion term)
    laplacian_omega = A @ omega0

    # Time derivative of omega
    wt = nu * laplacian_omega - J
    return wt

# Method using solve
def rhs_solve(t, omega0):
    psi = solve(A, omega0)
    dpsi_dx = B @ psi
    dpsi_dy = C @ psi
    domega_dx = B @ omega0
    domega_dy = C @ omega0
    J = dpsi_dx * domega_dy - dpsi_dy * domega_dx  # Element-wise product

    # Compute Laplacian of omega (diffusion term)
    laplacian_omega = A @ omega0

    # Time derivative of omega
    wt = nu * laplacian_omega - J
    return wt

# Method using LU
def rhs_LU(t, omega):

    # Solve A phi = omega using LU decomposition
    # Step 1: Solve L y = P omega (forward substitution)
    omega_permuted = P @ omega  # Apply permutation matrix to omega
    y = solve_triangular(L, omega_permuted, lower=True)  # Solve for y

    # Step 2: Solve U phi = y (backward substitution)
    psi = solve_triangular(U, y)  # Solve for phi

    dpsi_dx = B @ psi # Apply B to phi (partial_x phi)
    dpsi_dy = C @ psi # Apply C to phi (partial_y phi)
    domega_dx = B @ omega  # Apply B to omega (partial_x omega)
    domega_dy = C @ omega  # Apply C to omega (partial_y omega)

    J = dpsi_dx * domega_dy - dpsi_dy * domega_dx  # Jacobian term

    # Compute the Laplacian of omega (diffusion term)
    laplacian_omega = A @ omega

    # Time derivative of omega: advection + diffusion
    wt = nu * laplacian_omega - J

    return wt

def rhs_bicgstab(t, omega):
    # Solve A phi = omega using BICGSTAB
    # The BICGSTAB solver returns phi and info; we check info for convergence
    psi, info = bicgstab(A, omega, rtol = 0.5)
    
    if info != 0:
        print("Warning: BiCGSTAB did not converge")

    # Compute the Jacobian J(phi, omega) if needed for advection
    dpsi_dx = B @ psi.flatten()  # partial_x phi
    dpsi_dy = C @ psi.flatten()  # partial_y phi
    domega_dx = B @ omega  # partial_x omega
    domega_dy = C @ omega  # partial_y omega

    J = dpsi_dx * domega_dy - dpsi_dy * domega_dx  # Jacobian term

    # Compute the Laplacian of omega (diffusion term)
    laplacian_omega = A @ omega

    # Time derivative of omega: advection + diffusion
    wt = nu * laplacian_omega - J

    return wt

def rhs_gmres(t, omega):
    # Solve A phi = omega using GMRES
    # The GMRES solver returns phi and info; we check info for convergence
    psi, info = gmres(A, omega, rtol=0.25)
    
    if info != 0:
        print("Warning: GMRES did not converge")

    # Compute the Jacobian J(phi, omega) if needed for advection
    dpsi_dx = B @ psi  # partial_x phi
    dpsi_dy = C @ psi  # partial_y phi
    domega_dx = B @ omega  # partial_x omega
    domega_dy = C @ omega  # partial_y omega

    J = dpsi_dx * domega_dy - dpsi_dy * domega_dx  # Jacobian term

    # Compute the Laplacian of omega (diffusion term)
    laplacian_omega = A @ omega

    # Time derivative of omega: advection + diffusion
    wt = nu * laplacian_omega - J

    return wt

# def update_multi(frame):
#     for contour, sol, ax in zip(contours, solutions, axs):
#         # Clear the previous frame's contours
#         for c in contour[0].collections:
#             c.remove()
        
#         # Update the contour plot for the current frame
#         omega_frame = sol.y[:, frame].reshape((n, n))
#         contour[0] = ax.contourf(X, Y, omega_frame, levels=100, cmap=cmapc, alpha=0.9)

def create_animation(sol, X, Y, n, cmap, levels=50, interval=100, 
                     colorbar_ticks=5, axis_ticks=5, title="Vortex Dynamics"):
    """
    Create a contour animation for a given solution with customizations.

    Parameters:
    - sol: solve_ivp solution object containing `t` (time steps) and `y` (solution array).
    - X, Y: Meshgrid arrays for spatial coordinates.
    - n: Grid resolution (used for reshaping solution vectors).
    - cmap: Colormap for the contour plot.
    - levels: Number of contour levels (default: 50).
    - interval: Interval between frames in milliseconds (default: 100).
    - colorbar_ticks: Number of ticks for the colorbar (default: 5).
    - axis_ticks: Number of ticks for x and y axes (default: 5).
    - title: Custom title for the plot (default: "Vortex Dynamics").
    """
    fig, ax = plt.subplots()
    
    def update(frame):
        """Update the contour plot for the animation."""
        ax.clear()  # Clear the current frame
        omega = sol.y[:, frame].reshape((n, n))  # Reshape solution for current frame
        contour = ax.contourf(X, Y, omega, levels=levels, cmap=cmap)
        ax.set_title(f"{title} | Time: {sol.t[frame]:.2f}", fontsize=14)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=5))  # Adjust 'nbins' for number of ticks
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=5))
        return contour

    # Create the initial contour plot
    omega0 = sol.y[:, 0].reshape((n, n))
    contour = ax.contourf(X, Y, omega0, levels=levels, cmap=cmap)
    
    # Customize colorbar
    cbar = fig.colorbar(contour, ax=ax, shrink=0.8)
    cbar.locator = MaxNLocator(nbins=5)  # 'nbins' controls the number of ticks on the colorbar
    cbar.update_ticks() 
    ax.set_title(f"{title} | Time: {sol.t[0]:.2f}", fontsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=5))  # Adjust 'nbins' for number of ticks
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=5))
    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(sol.t), interval=interval)
    return ani

# Constructing Soarse Matrices
m = 64   # N value in x and y directions
n = m * m  # total size of matrix
dn = 20/m

e0 = np.zeros((n, 1))  # vector of zeros
e1 = np.ones((n, 1))   # vector of ones
e2 = np.copy(e1)    # copy the one vector
e4 = np.copy(e0)    # copy the zero vector

for j in range(1, m+1):
    e2[m*j-1] = 0  # overwrite every m^th value with zero
    e4[m*j-1] = 1  # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

# Place diagonal elements
diagonals_A = [e1.flatten(), e1.flatten(), e5.flatten(), 
             e2.flatten(), -4 * e1.flatten(), e3.flatten(), 
             e4.flatten(), e1.flatten(), e1.flatten()]
offsets_A = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]

A = spdiags(diagonals_A, offsets_A, n, n).toarray()
A = A/(dn**2)

diagonals_B = [e1.flatten(), e1.flatten(), -e1.flatten(), -e1.flatten()]
offsets_B = [m, -(n-m), (n-m), -m]
B = spdiags(diagonals_B, offsets_B, n, n).toarray()
B = B/(2 * dn)

diagonals_C = [e5.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets_C = [-(m-1), -1, 1, (m-1)] 

# Construct matrix C for partial derivative in the y-direction
C = spdiags(diagonals_C, offsets_C, n, n).toarray()
C = C / (2 * dn)

# Defining Parameters
l = 20
n = 64
nu = 0.001
tspan = np.arange(0, 4.5, 0.5)
x2 = np.linspace(-l/2, l/2, n+1)
x = x2[:n]
y = x2[:n]
dx = l/n
X, Y = np.meshgrid(x,y)

# FFT
kx = (2 * np.pi / l) * np.concatenate((np.arange(0, n/2), np.arange(-n/2, 0)))
ky = (2 * np.pi / l) * np.concatenate((np.arange(0, n/2), np.arange(-n/2, 0)))
kx[0] = 1e-6
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
k_squared = KX**2 + KY**2
omega0 = np.exp(-X**2 - Y**2 / 20).flatten()
start_time = time.time() # Record the start time
sol1 = solve_ivp(rhs_fourier, (tspan[0], tspan[-1]), omega0, t_eval=tspan, method='RK45')
end_time = time.time() # Record the end time
elapsed_time = end_time - start_time
print(f"FFT time: {elapsed_time:.2f} seconds")

# A\b
A[0,0] = 2
start_time = time.time() # Record the start time
sol2 = solve_ivp(rhs_solve, (tspan[0], tspan[-1]), omega0, t_eval=tspan, method='RK45')
end_time = time.time() # Record the end time
elapsed_time = end_time - start_time
print(f"A/b time: {elapsed_time:.2f} seconds")

# LU Decomposition
start_time = time.time() # Record the start time
P, L, U = lu(A)
sol3 = solve_ivp(rhs_LU, (tspan[0], tspan[-1]), omega0, t_eval=tspan, method='RK45')
end_time = time.time() # Record the end time
elapsed_time = end_time - start_time
print(f"LU time: {elapsed_time:.2f} seconds")

# BICGSTAB
start_time = time.time() # Record the start time
sol4 = solve_ivp(rhs_bicgstab, (tspan[0], tspan[-1]), omega0, t_eval=tspan, method='RK45')
end_time = time.time() # Record the end time
elapsed_time = end_time - start_time
print(f"BICGSTAB time: {elapsed_time:.2f} seconds")

# GMRES
start_time = time.time() # Record the start time
sol5 = solve_ivp(rhs_gmres, (tspan[0], tspan[-1]), omega0, t_eval=tspan, method='RK45')
end_time = time.time() # Record the end time
elapsed_time = end_time - start_time
print(f"GMRES time: {elapsed_time:.2f} seconds")

'''
Plot
'''
# num_plots = 6
# indices = np.linspace(0, len(tspan) - 1, num_plots, dtype=int)
# fig, axes = plt.subplots(2, 3, figsize=(10, 5))
# axes = axes.flatten()

# for i, ax in enumerate(axes):
#     idx = indices[i]
#     omega = sol1.y[:, idx].reshape(n, n)
#     ax.contourf(X, Y, omega, levels=50, cmap='jet')
#     ax.set_title(f"t = {tspan[idx]:.1f}")  # Use evenly spaced time indices
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')

# plt.tight_layout()
# plt.show()

'''
Solving different initial conditions using FFT
'''
# Two Oppositely Charged Gaussian Vorticies Next to Each Other
tspan = np.arange(0, 20.2, 0.2)
x_center1, y_center1 = -2, 0
x_center2, y_center2 = 2, 0
amplitude1, amplitude2 = -5, 5

omega0_a = (amplitude1 * np.exp(-(X-x_center1)**2 - (Y-y_center1)**2 / 20) + 
          amplitude2 * np.exp(-(X-x_center2)**2 - (Y-y_center2)**2 / 20)).flatten()
# sol_a = solve_ivp(rhs_fourier, (tspan[0], tspan[-1]), omega0_a, t_eval=tspan, method='RK45')

# Two Same Charged Gaussian Vorticies Next to Each Other
x_center1, y_center1 = -1.5, 0
x_center2, y_center2 = 1.5, 0
amplitude1, amplitude2 = 3, 3
omega0_b = (amplitude1*np.exp(-(X-x_center1)**2 - (Y-y_center1)**2 / 20) + 
            amplitude1*np.exp(-(X-x_center2)**2 - (Y-y_center2)**2 / 20)).flatten()
# sol_b = solve_ivp(rhs_fourier, (tspan[0], tspan[-1]), omega0_b, t_eval=tspan, method='RK45')

# Two Pairs of Oppositely Charged Vorticies made to collide
x_center1, y_center1 = -5, -1  # First pair, positive vortex
x_center2, y_center2 = -5, 1  # First pair, negative vortex
x_center3, y_center3 = 5, 1   # Second pair, negative vortex
x_center4, y_center4 = 5, -1  # Second pair, positive vortex
amplitude1, amplitude2 = -10, 10         # Opposite amplitudes for each pair

# Define the initial vorticity field with two pairs of oppositely charged Gaussian vortices
omega0_c = (
    amplitude2 * np.exp(-((X - x_center1)**2 + (Y - y_center1)**2) / 2) +
    amplitude1 * np.exp(-((X - x_center2)**2 + (Y - y_center2)**2) / 2) +
    amplitude2 * np.exp(-((X - x_center3)**2 + (Y - y_center3)**2) / 2) +
    amplitude1 * np.exp(-((X - x_center4)**2 + (Y - y_center4)**2) / 2)
).flatten()
# sol_c = solve_ivp(rhs_fourier, (tspan[0], tspan[-1]), omega0_c, t_eval=tspan, method='RK45')

# Random Assortment of Vorticies
num_vortices = 15
# Initialize omega0 with random vortices
omega0_d = np.zeros_like(X)

for _ in range(num_vortices):
    # Random properties for each vortex
    x_center = np.random.uniform(-l/2.5, l/2.5)  # Center in X within domain
    y_center = np.random.uniform(-l/2.5, l/2.5)  # Center in Y within domain
    amplitude = np.random.uniform(-5, 5)     # Random charge/strength
    width_x = np.random.uniform(0.5, 1.5)    # Ellipticity in x
    width_y = np.random.uniform(0.5, 1.5) 
    
    # Add this vortex to the initial condition
    vortex = amplitude * np.exp(-((X - x_center)**2 / (2 * width_x**2) + 
                                  (Y - y_center)**2 / (2 * width_y**2)))
    omega0_d += vortex

# Solve the differential equation with the random initial condition
omega0_d = omega0_d.flatten()
sol_d = solve_ivp(rhs_fourier, (tspan[0], tspan[-1]), omega0_d, t_eval=tspan, method='RK45')

'''
Animations
'''

colors = ["#091680",
          "#1FADFF",  
          "#33f6ff",  
          "#FFFFFF", 
          "#FF67E1", 
          "#A800AB",
          "#4D0687"] 


# Create the colormap
cmapc = LinearSegmentedColormap.from_list("navy_to_purple", colors, N=256)
# Separate Plots
# ani1 = create_animation(
#     sol_a, X, Y, n, cmapc, levels=100, interval=100, 
#     colorbar_ticks=5, axis_ticks=5, title='Oppositely Charged Vorticies'
# )
# ani2 = create_animation(
#     sol_b, X, Y, n, cmapc, levels=100, interval=100, 
#     colorbar_ticks=5, axis_ticks=5, title='Same Charged Vorticies'
# )
# ani3 = create_animation(
#     sol_c, X, Y, n, cmapc, levels=100, interval=100, 
#     colorbar_ticks=5, axis_ticks=5, title='Colliding Vorticies'
# )
ani4 = create_animation(
    sol_d, X, Y, n, cmapc, levels=100, interval=100, 
    colorbar_ticks=5, axis_ticks=5, title='Random Vorticies'
)
# ani4.save("Random Vorticies.mp4", writer="ffmpeg", fps=30)
plt.show()





# Same Screen Animations
# fig, axs = plt.subplots(2, 2, figsize=(10, 7))
# axs = axs.flatten()
# # Plot each initial condition and prepare contours for updating
# contours = []
# solutions = [sol_a, sol_b, sol_c, sol_d]
# initial_conditions = [omega0_a, omega0_b, omega0_c, omega0_d]
# titles = ['Oppositely Charged Vorticies', 'Same Charged Vorticies', 
#           'Colliding Vorticies', 'Random Vorticies']

# for i, ax in enumerate(axs):
#     ax.set_title(titles[i], fontsize=12)
#     contour = ax.contourf(X, Y, initial_conditions[i].reshape((n, n)), 
#                           levels=100, cmap=cmapc, alpha=0.9)
#     contours.append([contour])
#     cbar = fig.colorbar(contour, ax=ax, shrink=0.8) # Make colorbar tick labels white
    
#     # Reduce the number of ticks on the colorbar using MaxNLocator
#     cbar.locator = MaxNLocator(nbins=5)  # 'nbins' controls the number of ticks on the colorbar
#     cbar.update_ticks() 
#     # Reduce the number of ticks on both axes
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=5))  # Adjust 'nbins' for number of ticks
#     ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=5))
# # Create the animation
# ani = FuncAnimation(fig, update_multi, frames=len(sol_a.t), interval=100)

# # Show the animation
# plt.tight_layout()
# plt.show()