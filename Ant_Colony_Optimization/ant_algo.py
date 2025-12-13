import numpy as np
import matplotlib.pyplot as plt
import string

# -----------------------------
# Utility functions
# -----------------------------
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def compute_distance_matrix(points):
    n = len(points)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = euclidean_distance(points[i], points[j])
    return dist


# -----------------------------
# Ant Colony Optimization
# -----------------------------
def ant_colony_optimization(
    points,
    n_ants=30,
    n_iterations=100,
    alpha=1.0,        # pheromone importance
    beta=5.0,         # heuristic importance
    evaporation=0.5,
    Q=100
):
    n = len(points)
    dist = compute_distance_matrix(points)
    pheromone = np.ones((n, n))

    best_path = None
    best_length = np.inf
    history = []

    for iteration in range(n_iterations):
        all_paths = []
        all_lengths = []

        for ant in range(n_ants):
            visited = [False] * n
            start = np.random.randint(n)
            path = [start]
            visited[start] = True

            while len(path) < n:
                current = path[-1]
                probabilities = []

                for j in range(n):
                    if not visited[j]:
                        tau = pheromone[current][j] ** alpha
                        eta = (1.0 / dist[current][j]) ** beta
                        probabilities.append(tau * eta)
                    else:
                        probabilities.append(0)

                probabilities = np.array(probabilities)
                probabilities /= probabilities.sum()

                next_city = np.random.choice(range(n), p=probabilities)
                path.append(next_city)
                visited[next_city] = True

            path.append(start)
            length = sum(dist[path[i]][path[i+1]] for i in range(len(path)-1))

            all_paths.append(path)
            all_lengths.append(length)

            if length < best_length:
                best_length = length
                best_path = path

        # Evaporation
        pheromone *= (1 - evaporation)

        # Reinforcement
        for path, length in zip(all_paths, all_lengths):
            for i in range(len(path)-1):
                pheromone[path[i]][path[i+1]] += Q / length

        # Elitism
        for i in range(len(best_path)-1):
            pheromone[best_path[i]][best_path[i+1]] += Q / best_length

        history.append(best_length)

        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Best Length: {best_length:.2f}")

    return best_path, best_length, pheromone, history


# -----------------------------
# Visualization Functions
# -----------------------------
def plot_best_path(points, path, city_names):
    plt.figure(figsize=(8, 8))

    for i in range(len(path)-1):
        a = points[path[i]]
        b = points[path[i+1]]
        plt.plot([a[0], b[0]], [a[1], b[1]],
                 color='blue', linewidth=2)

    plt.scatter(points[:, 0], points[:, 1],
                c='red', s=120, zorder=5)

    for i, p in enumerate(points):
        plt.text(p[0] + 0.02, p[1] + 0.02,
                 city_names[i],
                 fontsize=12,
                 fontweight='bold')

    plt.title("Best Tour using ACO")
    plt.grid(True)
    plt.show()


def plot_pheromone_map(points, pheromone, city_names):
    plt.figure(figsize=(8, 8))
    n = len(points)
    max_pher = pheromone.max()

    for i in range(n):
        for j in range(n):
            if pheromone[i][j] > 0.01:
                p1 = points[i]
                p2 = points[j]
                plt.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    linewidth=2 * pheromone[i][j] / max_pher,
                    alpha=pheromone[i][j] / max_pher,
                    color='green'
                )

    plt.scatter(points[:, 0], points[:, 1],
                c='black', s=120)

    for i, p in enumerate(points):
        plt.text(p[0] + 0.02, p[1] + 0.02,
                 city_names[i],
                 fontsize=12,
                 fontweight='bold')

    plt.title("Pheromone Intensity Map")
    plt.grid(True)
    plt.show()


def plot_convergence(history):
    plt.figure(figsize=(7, 4))
    plt.plot(history, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best Tour Length")
    plt.title("ACO Convergence")
    plt.grid(True)
    plt.show()


# -----------------------------
# Run Example
# -----------------------------
np.random.seed(42)
points = np.random.rand(12, 2)

city_names = list(string.ascii_uppercase[:len(points)])

best_path, best_length, pheromone, history = ant_colony_optimization(
    points,
    n_ants=40,
    n_iterations=120,
    alpha=1,
    beta=5,
    evaporation=0.4
)

print("\nFinal Best Path:", [city_names[i] for i in best_path])
print("Final Best Length:", best_length)

plot_best_path(points, best_path, city_names)
plot_pheromone_map(points, pheromone, city_names)
plot_convergence(history)
