import random
import math

# --- Problem Definitions ---

def get_bpp1_weights(k=500):
    """
    Problem BPP1: 
    - b=10 bins
    - weight of item i is i
    - Item indices begin at 1
    """
    return [i for i in range(1, k + 1)] # Weights from 1 to k+1

def get_bpp2_weights(k=500):
    """
    Problem BPP2: 
    - b=50 bins
    - weight of item i is (i^2)/2
    """
    return [(i**2) / 2 for i in range(1, k+1)] 

K_ITEMS = 500
BINS_BPP1 = 10
BINS_BPP2 = 50

# --- Chromosome and Fitness ---

def initialise_chromosome(k, num_bins):
    """
    Creates a random chromosome for BPP with k items and each gene is a rand int from 1 to num_bins.
    Using list as it allows direct acces to bin assign for each item
    """
    return [random.randint(1, num_bins) for _ in range(k)]

def evaluate_fitness(chromosome, weights, num_bins):
    """
    Calcs the fitness:
    - 100 / ( 1 + d ), d is the diff between the heaviest and lightest bin.
    """

    # init num_bins to 0
    bin_totals = {bin_id: 0 for bin_id in range(1, num_bins + 1)} # using a dict as it is more efficient and clear

    # Iterate through each item and add its weight to the appropriate bin
    for item_index, bin_assignment in enumerate(chromosome):
        weight = weights[item_index]
        bin_totals[bin_assignment] += weight
    
    # get the difference between heaviest and lightest bin
    if not bin_totals:
        return 0.0  # Avoid division by zero if no bins
    
    total_values = list(bin_totals.values())
    
    heaviest = max(total_values)
    lightest = min(total_values)

    d = heaviest - lightest  # The metric to minimise (d)

    # Calculate and Return Fitness
    # formula: fitness = 100 / (1 + d)
    # This converts MINIMISATION of d into maximise fitness,
    # which is what GA should aim for.

    fitness = 100.0 / (1.0 + d)
    
    return fitness, d

def tournament_selection(population, fitnesses, tournament_size):
    """
    Selects a parent using tournament selection of sixe t
    fitness must be maximised
    """

    # randomly select t chromosomes
    tournament_indices = random.sample(range(len(population)), tournament_size)

    best_index = -1
    best_fitness = -1.0

    # choose the chromosome with the best fitness from this tournament
    for index in tournament_indices:
        if fitnesses[index] > best_fitness:
            best_fitness = fitnesses[index]
            best_index = index
            
    return population[best_index]

def uniform_crossover(parent1, parent2, pc=0.8):
    """
    Implements uniform crossover with probability pc.
    If crossover does not occur one parent is selected as the offspring.
    """
    if random.random() < pc:
        # Crossover happens:
        offspring = []
        for gene1, gene2 in zip(parent1, parent2):
            # 50/50 chance
            if random.random() < 0.5:
                # chance 1, take from parent 1
                offspring.append(gene1)
            else:
                # chance 2 , take from parent 2
                offspring.append(gene2)
        return offspring
    else:
        # Crossover does not occur so return one of the parents 
        return list(parent1) 

def random_reassignment_mutation(chromosome, pm, num_bins):
    """
    Mutates the chromosome by randomly reassigning a gene's bin with probability pm.
    """
    mutated_chromosome = list(chromosome) # Create a mutable copy
    
    for i in range(len(mutated_chromosome)):
        # Check for mutation probability
        if random.random() < pm:
            # Randomly change the bin assignment to any valid bin 
            mutated_chromosome[i] = random.randint(1, num_bins)
            
    return mutated_chromosome

