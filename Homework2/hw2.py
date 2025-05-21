import random

cities = [
    (0, 3), (0, 0),
    (0, 2), (0, 1),
    (1, 0), (1, 3),
    (2, 0), (2, 3),
    (3, 0), (3, 3),
    (3, 1), (3, 2)
]

def hill_climbing(initial_path, cost_fn, neighbor_fn, max_attempts=10000):
    current_path = initial_path
    current_cost = cost_fn(current_path)
    print("Initial cost:", current_cost, current_path)

    attempts = 0
    while attempts < max_attempts:
        candidate = neighbor_fn(current_path)
        candidate_cost = cost_fn(candidate)

        if candidate_cost < current_cost:
            current_path = candidate
            current_cost = candidate_cost
            attempts = 0
            print("New best:", current_cost, current_path)
        else:
            attempts += 1

    print("Final solution:", current_cost, current_path)
    return current_path

def calculate_total_distance(path):
    total = 0
    for i, city_index in enumerate(path):
        next_index = path[(i + 1) % len(path)]
        total += euclidean_distance(cities[city_index], cities[next_index])
    return total

def generate_neighbor(path):
    new_path = path[:]
    i, j = sorted(random.sample(range(len(path)), 2))
    new_path[i:j+1] = reversed(new_path[i:j+1])
    return new_path

def euclidean_distance(a, b):
    (x1, y1), (x2, y2) = a, b
    return ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

initial = list(range(len(cities)))
best_result = hill_climbing(initial, calculate_total_distance, generate_neighbor)
