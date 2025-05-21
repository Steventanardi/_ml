import numpy as np

def f(x, y, z):
    return x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

def hill_climb(f, start, step_size=0.1, max_iter=1000, tol=1e-6):
    current = np.array(start)
    for _ in range(max_iter):
        neighbors = [current + step_size * np.eye(3)[i] for i in range(3)] + \
                    [current - step_size * np.eye(3)[i] for i in range(3)]
    
        next_point = min(neighbors, key=lambda p: f(*p))
        if f(*next_point) < f(*current) - tol:
            current = next_point
        else:
            break
    return current, f(*current)

start = [0.0, 0.0, 0.0]
opt_point, opt_value = hill_climb(f, start)

print("Minimum point found at: x = %.4f, y = %.4f, z = %.4f" % tuple(opt_point))
print("Minimum value f(x, y, z) = %.4f" % opt_value)
