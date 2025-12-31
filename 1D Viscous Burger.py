import deepxde as dde
import numpy as np

# Spatial domain: Line from -1 to 1
geom = dde.geometry.Interval(-1, 1)

# Time domain: 0 to 1
timedomain = dde.geometry.TimeDomain(0, 1)

# Combine them into a 2D Spacetime domain (Rectangle)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

def pde(x, y):
    # dy_t is du/dt
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)

    # dy_x is du/dx
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)

    # dy_xx is d^2u/dx^2
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)

    # Residual: du/dt + u*du/dx - v*d^2u/dx^2 = 0
    return dy_t + y * dy_x - (0.01 / np.pi) * dy_xx

# Boundary Condition: u=0 at x=-1 and x=1
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)

# Initial Condition: u = -sin(pi*x) at t=0
ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)

data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=2500, num_boundary=100, num_initial=100)
net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
model.train(iterations=15000)

model.compile("L-BFGS")
model.train()
