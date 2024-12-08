# Vortex Streamfunction

This repository contains Python code to simulate and analyze the dynamics of vortices in a 2D domain. Various numerical methods are implemented to solve the vorticity-streamfunction equations, with results visualized using contour plots and animations.

---

## Features

- **Four Numerical Methods for Solving Poisson Equation**:
  - Fourier Transform
  - Direct Solve (numpy.linalg.solve)
  - LU Decomposition
  - Iterative Methods (BiCGSTAB, GMRES)

- **Initial Conditions**:
  - Single Gaussian vortex.
  - Two oppositely charged Gaussian vortices.
  - Two same-charged Gaussian vortices.
  - Colliding vortex pairs.
  - Randomly generated vortex field.

- **Visualization**:
  - Contour plots of the vorticity field over time.
  - Animated evolution of vortex dynamics.

---

## Mathematical Formulation

The time evolution of the voritcity $$\omega(x,y,t)$$ and streamfunction $$\psi(x,y,t)$$ which satisfies $$\nabla^2\psi=\omega$$ (Poisson Equation) are given by the governing equations

$$
\omega_t+[\psi,\omega]=\nu\nabla^2\omega
$$

Where:
- $$[\psi,\omega]=\psi_x\omega_y-\psi_y\omega_x$$
- $$\nabla^2=\partial_x^2+\partial_y^2$$

The numerical solution involves discretizing the domain and solving these equations iteratively using various numerical methods.

---

## Code Overview

### Key Functions

1. **`rhs_fourier`**: Uses Fast Fourier Transform (FFT) to compute $$\psi$$ from $$\omega$$.
2. **`rhs_solve`**: Solves $$\nabla^2 \psi = \omega$$ directly using matrix inversion.
3. **`rhs_LU`**: Solves the Poisson equation using LU decomposition.
4. **`rhs_bicgstab`**: Uses BiCGSTAB to solve the Poisson equation iteratively.
5. **`rhs_gmres`**: Uses GMRES to solve the Poisson equation iteratively.

### Visualization

- **Contour Animations**:
  Animations of the vorticity field evolution are generated using `matplotlib.animation`. These provide insights into the interaction and evolution of vortices over time.

---

## Results

### Performance Comparison

The time taken by each numerical method to solve the Poisson equation for a single Gaussian vortex is displayed in the console output.

### Example Plots and Animations

#### Single Gaussian Vortex:
- Initial condition: $$\omega(x, y) = e^{-x^2 - y^2 / 20}$$

#### Two Oppositely Charged Gaussian Vortices:
- See videos folder

#### Two Same-Charged Gaussian Vortices:
- See videos folder
  
#### Colliding Vortex Pairs:
- See videos folder
  
#### Randomly Generated Vortices:
- See videos folder
