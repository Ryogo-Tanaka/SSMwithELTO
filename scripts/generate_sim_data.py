import numpy as np
import os

def simulate_linear_nonlinear(T, A, Q, g, sigma, x0=None):
    r = A.shape[0]
    if x0 is None:
        x0 = np.zeros(r)
    xs = np.zeros((T, r))
    d = g(x0).shape[0]
    ds = np.zeros((T, d))
    x = x0.copy()
    for t in range(T):
        w = np.random.multivariate_normal(np.zeros(r), Q)
        x = A @ x + w
        ds[t] = g(x) + np.random.randn(d) * sigma
        xs[t] = x
    return xs, ds

def g_simple(x):
    # (1) x1, x1^2, x2, x2^2, x3, x3^2
    return np.array([x[0], x[0]**2, x[1], x[1]**2, x[2], x[2]**2])

def g_complex(x):
    # (2) sin, exp, tanh, cross terms...
    return np.array([
        np.sin(x[0]),
        x[1]**2,
        np.exp(-x[2]),
        x[0]*x[1],
        np.tanh(x[2]),
        x[0] + x[1] + x[2]
    ])

def main():
    os.makedirs("data", exist_ok=True)
    T = 10000
    A = np.array([[0.8,0.1,0.0],
                  [0.0,0.9,0.1],
                  [0.0,0.0,0.7]])
    Q = 0.01 * np.eye(3)
    sigma = 0.05

    # シンプル非線形観測
    xs1, ds1 = simulate_linear_nonlinear(T, A, Q, g_simple, sigma)
    np.savez("data/sim_simple.npz", X=xs1, Y=ds1)

    # 複雑非線形観測
    xs2, ds2 = simulate_linear_nonlinear(T, A, Q, g_complex, sigma)
    np.savez("data/sim_complex.npz", X=xs2, Y=ds2)

    # week noise
    sigma = 0.005
    xs3, ds3 = simulate_linear_nonlinear(T, A, Q, g_simple, sigma)
    np.savez("data/sim_simple_wkns.npz", X=xs3, Y=ds3)

    xs4, ds4 = simulate_linear_nonlinear(T, A, Q, g_complex, sigma)
    np.savez("data/sim_complex_wknz.npz", X=xs4, Y=ds4)

    #no noise
    sigma = 0
    xs5, ds5 = simulate_linear_nonlinear(T, A, Q, g_simple, sigma)
    np.savez("data/sim_simple_nonz.npz", X=xs5, Y=ds5)

    xs6, ds6 = simulate_linear_nonlinear(T, A, Q, g_complex, sigma)
    np.savez("data/sim_complex_nonz.npz", X=xs6, Y=ds6)

    print("Generated sim_simple.npz and sim_complex.npz in data/")

if __name__ == "__main__":
    main()
    # print("sim. finished!")
