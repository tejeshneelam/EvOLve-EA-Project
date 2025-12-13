from random import random, randint, seed      # 🎲 randomness for evolution
from statistics import mean                   # 📊 for fitness calculation
from copy import deepcopy                     # 🧬 safe copying of trees

# -------------------------
# GP CONFIGURATION
# -------------------------

POP_SIZE        = 60    # 👥 number of programs in population
MIN_DEPTH       = 2     # 🌱 minimum initial tree depth
MAX_DEPTH       = 5     # 🌳 maximum initial tree depth
GENERATIONS     = 250   # 🔁 max generations to evolve
TOURNAMENT_SIZE = 5     # ⚔️ selection pressure
XO_RATE         = 0.8   # 🔀 crossover probability
PROB_MUTATION   = 0.2   # 🧪 mutation probability per node

# -------------------------
# FUNCTION SET
# -------------------------
# Internal nodes of GP tree

def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y

FUNCTIONS = [add, sub, mul]   # ➕ ➖ ✖️

# -------------------------
# TERMINAL SET
# -------------------------
# Leaf nodes of GP tree

TERMINALS = ['x', -2, -1, 0, 1, 2]   # 📥 input variable + constants

# -------------------------
# TARGET FUNCTION
# -------------------------
# GP tries to rediscover this function only using data points

def target_func(x):
    return x*x*x*x + x*x*x + x*x + x + 1   # 🎯 polynomial target

# -------------------------
# DATASET GENERATION
# -------------------------
# Create (x, y) pairs for symbolic regression

def generate_dataset():
    dataset = []
    for x in range(-100, 101, 2):  # values from -1.0 to +1.0
        x /= 100
        dataset.append([x, target_func(x)])
    return dataset

# ============================================================
# GENETIC PROGRAMMING TREE CLASS
# ============================================================

class GPTree:
    def __init__(self, data=None, left=None, right=None):
        self.data  = data     # 📌 function or terminal
        self.left  = left     # ⬅️ left subtree
        self.right = right    # ➡️ right subtree

    # -------------------------
    # Node label for printing
    # -------------------------
    def node_label(self):
        if self.data in FUNCTIONS:
            return self.data.__name__  # function name
        else:
            return str(self.data)      # terminal value

    # -------------------------
    # Print tree structure
    # -------------------------Preorder traversal
    def print_tree(self, prefix=""):
        print(prefix + self.node_label())
        if self.left:
            self.left.print_tree(prefix + "   ")
        if self.right:
            self.right.print_tree(prefix + "   ")

    # -------------------------
    # Execute the program tree
    # -------------------------Postorder evaluation
    def compute_tree(self, x):
        if self.data in FUNCTIONS:     # 🧠 internal node
            return self.data(
                self.left.compute_tree(x),
                self.right.compute_tree(x)
            )
        elif self.data == 'x':         # 📥 input variable
            return x
        else:                          # 🔢 constant
            return self.data

    # -------------------------
    # Random tree generation
    # uses Grow / Full methods
    # -------------------------
    def random_tree(self, grow, max_depth, depth=0):
        if depth < MIN_DEPTH:
            self.data = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]
        elif depth >= max_depth:
            self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
        else:
            if grow and random() < 0.5:
                self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
            else:
                self.data = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]

        if self.data in FUNCTIONS:
            self.left = GPTree()
            self.left.random_tree(grow, max_depth, depth + 1)
            self.right = GPTree()
            self.right.random_tree(grow, max_depth, depth + 1)

    # -------------------------
    # Mutation operator
    # -------------------------
    def mutation(self):
        if random() < PROB_MUTATION:   # 🧪 random subtree replacement
            self.random_tree(grow=True, max_depth=2)
        elif self.left:
            self.left.mutation()
        elif self.right:
            self.right.mutation()

    # -------------------------
    # Tree size (for bloat)
    # -------------------------
    def size(self):
        if self.data in TERMINALS:
            return 1
        return 1 + (self.left.size() if self.left else 0) + \
                   (self.right.size() if self.right else 0)

    # -------------------------
    # Copy subtree
    # -------------------------
    def build_subtree(self):
        t = GPTree(self.data)
        if self.left:
            t.left = self.left.build_subtree()
        if self.right:
            t.right = self.right.build_subtree()
        return t

    # -------------------------
    # Locate subtree for crossover
    # -------------------------
    def scan_tree(self, count, second):
        count[0] -= 1
        if count[0] <= 1:
            if not second:
                return self.build_subtree()
            else:
                self.data = second.data
                self.left = second.left
                self.right = second.right
        else:
            if self.left:
                self.left.scan_tree(count, second)
            if self.right:
                self.right.scan_tree(count, second)

    # -------------------------
    # Crossover operator
    # -------------------------
    def crossover(self, other):
        if random() < XO_RATE:   # 🔀 subtree swap
            second = other.scan_tree([randint(1, other.size())], None)
            self.scan_tree([randint(1, self.size())], second)

# ============================================================
# INITIAL POPULATION (RAMPED HALF-AND-HALF)
# ============================================================

def init_population():
    population = []
    for depth in range(3, MAX_DEPTH + 1):
        for _ in range(POP_SIZE // 6):
            t = GPTree()
            t.random_tree(grow=True, max_depth=depth)
            population.append(t)
        for _ in range(POP_SIZE // 6):
            t = GPTree()
            t.random_tree(grow=False, max_depth=depth)
            population.append(t)
    return population

# ============================================================
# FITNESS FUNCTION
# ============================================================

def fitness(individual, dataset):
    # 🎯 inverse mean absolute error (normalized to 0–1)
    return 1 / (1 + mean(
        abs(individual.compute_tree(x) - y) for x, y in dataset
    ))

# ============================================================
# TOURNAMENT SELECTION
# ============================================================

def selection(population, fitnesses):
    contenders = [randint(0, len(population)-1) for _ in range(TOURNAMENT_SIZE)]
    best = max(contenders, key=lambda i: fitnesses[i])
    return deepcopy(population[best])

# ============================================================
# MAIN EVOLUTION LOOP
# ============================================================

def main():
    seed()                                # 🎲 init RNG
    dataset = generate_dataset()          # 📊 training data
    population = init_population()        # 🌱 initial population

    best_of_run = None
    best_of_run_f = 0
    best_of_run_gen = 0

    fitnesses = [fitness(ind, dataset) for ind in population]

    for gen in range(GENERATIONS):
        next_population = []

        for _ in range(POP_SIZE):
            p1 = selection(population, fitnesses)
            p2 = selection(population, fitnesses)
            p1.crossover(p2)
            p1.mutation()
            next_population.append(p1)

        population = next_population
        fitnesses = [fitness(ind, dataset) for ind in population]

        if max(fitnesses) > best_of_run_f:
            best_of_run_f = max(fitnesses)
            best_of_run_gen = gen
            best_of_run = deepcopy(population[fitnesses.index(best_of_run_f)])

            print("________________________")
            print("gen:", gen, " best fitness:", round(best_of_run_f, 3))
            best_of_run.print_tree()

        if best_of_run_f == 1:
            break

    print("\nEND OF RUN")
    print("Best solution at generation:", best_of_run_gen)
    best_of_run.print_tree()

# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
