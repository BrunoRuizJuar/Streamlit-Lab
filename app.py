import numpy as np

class GA:
    def __init__(
        self,
        fitness_fn,
        pop_size=30,
        num_genes=10,
        crossover_rate=0.8,
        mutation_rate=0.1,
        generations=50,
        seed=0
    ):
        self.fitness_fn = fitness_fn
        self.pop_size = pop_size
        self.num_genes = num_genes
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        
        np.random.seed(seed)

        # Población inicial en [0,1]
        self.population = np.random.rand(pop_size, num_genes)

    # Selección por torneo
    def tournament_selection(self, k=3):
        idx = np.random.choice(self.pop_size, k)
        subset = self.population[idx]
        fitness_vals = [self.fitness_fn(ind) for ind in subset]
        return subset[np.argmax(fitness_vals)]

    # Crossover de un punto
    def crossover(self, p1, p2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.num_genes)
            return np.concatenate([p1[:point], p2[point:]])
        return p1.copy()

    # Mutación
    def mutate(self, individual):
        for i in range(self.num_genes):
            if np.random.rand() < self.mutation_rate:
                individual[i] = np.random.rand()
        return individual

    def run(self):
        best_history = []

        for _ in range(self.generations):
            new_pop = []
            for _ in range(self.pop_size):
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_pop.append(child)
            
            self.population = np.array(new_pop)

            # Guardar mejor fitness de esta generación
            best = max(self.population, key=self.fitness_fn)
            best_history.append(self.fitness_fn(best))

        return best, best_history
