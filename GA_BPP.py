import random
import numpy as np
import matplotlib.pyplot as plt

# --- Fixed Params ---
K_ITEMS = 500
BINS_BPP1 = 10
BINS_BPP2 = 50
PC = 0.8  
P_SIZE = 100

# --- Problem Definitions ---

def get_bpp1_weights(k=500):
    """
    Problem BPP1: 
    - b=10 bins
    - weight of item i is i (1 - 500)
    """
    return [i for i in range(1, k + 1)] # Weights from 1 to k+1 (500)

def get_bpp2_weights(k=500):
    """
    Problem BPP2: 
    - b=50 bins
    - weight of item i is (i^2)/2
    """
    return [(i**2) / 2 for i in range(1, k+1)] 

# --- Chromosome and Fitness ---

def initialise_chromosome(k, num_bins):
    """
    Create a list that shows where every item goes.
    Creates a random chromosome for BPP with k items and each gene is a rand int from 1 to num_bins.
    Using list as it allows direct acces to bin assign for each item
    [2, 3, 1, 2, 2, 3] This means: item 1 -> bin 2, item 2 -> bin 3, item 3 -> bin 1, item 4 -> bin 2, item 5 -> bin 2, item 6 -> bin 3
    """
    return [random.randint(1, num_bins) for _ in range(k)]

def evaluate_fitness(chromosome, weights, num_bins):
    """
    Calculate the difference (d) and the final fitness value:
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
    # This converts minimisation of d into maximise fitness,
    fitness = 100.0 / (1.0 + d)
    
    return fitness, d

def tournament_selection(population, fitnesses, tournament_size):
    """
    Selects the "fitter" parent using tournament selection of sixe t
    fitness must be maximised
    """

    # 1. Randomly select t chromosomes from the population (tournament size)
    tournament_indices = random.sample(range(len(population)), tournament_size)

    best_index = -1
    best_fitness = -1.0

    #2. Choose the chromosome with the best fitness from this tournament
    for index in tournament_indices:
        if fitnesses[index] > best_fitness:
            best_fitness = fitnesses[index]
            best_index = index
            
    return population[best_index]

def uniform_crossover(parent1, parent2, pc):
    """
    Combine parents to mix traits
    Implements uniform crossover with probability pc.
    If crossover does not occur one parent is selected as the offspring.
    """
    if random.random() < pc:
        # Crossover happens:
        offspring = []
        for gene1, gene2 in zip(parent1, parent2):
            #1. For each gene position, flip a coin
            if random.random() < 0.5:
                # If heads, take gene from parent 1
                offspring.append(gene1)
            else:
                # if tails, take gene from parent 2
                offspring.append(gene2)
        return offspring
    else:
        # Crossover does not occur so return one of the parents 
        return list(parent1) 

def random_reassignment_mutation(chromosome, pm, num_bins):
    """
    Introduce random genetic variation
    Mutates the chromosome by randomly reassigning a genes bin with probability pm.
    """
    mutated_chromosome = list(chromosome)
    # 1. For each gene, with probability pm (mutation rate)
    for i in range(len(mutated_chromosome)):
        # Check for mutation probability
        if random.random() < pm:
            # 2. Randomly change the bin assignment to any valid bin (1 to b)
            mutated_chromosome[i] = random.randint(1, num_bins)
            
    return mutated_chromosome

def run_ga(p, pm, t_size, num_bins, weights, seed, pc):
    """
    Runs a single trial of the GA with specified parameters and seed.
    Returns the best fitness, the corresponding d, and the convergence data.
    """
    random.seed(seed) # Sets the seed for reproducibility

    # Init Population and Variables
    population = [initialise_chromosome(K_ITEMS, num_bins) for _ in range(p)]
    
    # Evaluate initial population
    fitness_results = [evaluate_fitness(c, weights, num_bins) for c in population]
    fitnesses = [res[0] for res in fitness_results]
    
    evaluations = p # Start counter after initial pop evaluation
    
    # Track the best individual across all generations
    best_fitness_overall = max(fitnesses)
    best_d_overall = fitness_results[fitnesses.index(best_fitness_overall)][1]
    
    # Store convergence data to be graphed (Evaluation count, Best Fitness found)
    convergence_data = [(evaluations, best_fitness_overall)]
    
    # Main GA Loop
    while evaluations < 10000:
        # Find the Elite (best from current population)
        elite_index = fitnesses.index(max(fitnesses))
        elite_chromosome = population[elite_index]
        
        new_population = []
        
        # Create Offspring (Generational Replacement)
        # We need p offspring to fully replace the population 
        while len(new_population) < p:
            # Select Parents
            parent1 = tournament_selection(population, fitnesses, t_size)
            parent2 = tournament_selection(population, fitnesses, t_size)
            
            # Crossover
            offspring = uniform_crossover(parent1, parent2, pc)
            
            # Mutation
            mutated_offspring = random_reassignment_mutation(offspring, pm, num_bins)
            
            new_population.append(mutated_offspring)
            
        # Replace Population (with Elitism)
        
        # Ensure only the best chromosome is kept 
        # We replace the worst individual in the new population with the elite
        new_fitness_results = [evaluate_fitness(c, weights, num_bins) for c in new_population]
        new_fitnesses = [res[0] for res in new_fitness_results]
        
        # Update evaluations count
        evaluations += p 
        
        # Find the index of the worst offspring to replace with the elite
        worst_offspring_index = new_fitnesses.index(min(new_fitnesses))
        
        # Replace worst with elite Generational Replacement + Elitism
        new_population[worst_offspring_index] = elite_chromosome
        
        # Revaluate the population after elitism 
        population = new_population
        fitness_results = [evaluate_fitness(c, weights, num_bins) for c in population]
        fitnesses = [res[0] for res in fitness_results]

        # Track Best Overall
        current_best_fitness = max(fitnesses)
        current_best_d = fitness_results[fitnesses.index(current_best_fitness)][1]
        
        if current_best_fitness > best_fitness_overall:
            best_fitness_overall = current_best_fitness
            best_d_overall = current_best_d
            
        # Record data point for the graph
        if evaluations < 10000: # Check we have not hit the limit
             convergence_data.append((evaluations, best_fitness_overall))


    # Final recording at 10,000 evaluations
    convergence_data.append((10000, best_fitness_overall))
    
    return best_fitness_overall, best_d_overall, convergence_data


def run_experiment_set(problem_name, num_bins, weights, p, pm, t_size, n_trials=5, pc_rate=0.8):
    """
    Runs 5 trials for a specific parameter configuration.
    Returns a list of final results and the convergence data from all trials.
    """
    results = [] # Stores final_fitness and final_d for the 5 trials
    all_convergence_data = [] # Stores evaluations and best_fitness for all trials

    print(f"Running {problem_name} with pm={pm}, t={t_size}")

    for i in range(1, n_trials + 1):
        # use different seeds for each trial
        final_fitness, final_d, convergence_data = run_ga(p, pm, t_size, num_bins, weights, seed=i, pc=pc_rate)
        
        results.append({
            'trial': i,
            'seed': i,
            'final_fitness': final_fitness,
            'final_d': final_d
        })
        all_convergence_data.append(convergence_data)
        
    return results, all_convergence_data


def plot_convergence_curves(convergence_data_dict, problem_prefix, title, filename):
    """
    Processes the raw convergence data, calculates the
    mean fitness curve, and saves the plot.
    """
    plt.figure(figsize=(10, 6))
    
    # Define colours and styles for the 4 settings
    styles = {
        '_pm0.01_t3': ('b-', 'pm=0.01, t=3'),  # Blue solid
        '_pm0.05_t3': ('g:', 'pm=0.05, t=3'),  # Green dotted 
        '_pm0.01_t7': ('r--', 'pm=0.01, t=7 (Optimal)'), # Red dashed 
        '_pm0.05_t7': ('c-.', 'pm=0.05, t=7')  # Cyan dash-dot
    }

    # Standardize the X-axis for plotting as it is always the same
    fixed_evaluations = np.linspace(P_SIZE, 10000, int(10000/P_SIZE), dtype=int)

    # Plot the curves
    for key_suffix, (style, label) in styles.items():
        full_key = problem_prefix + key_suffix
        if full_key in convergence_data_dict:
            raw_trials = convergence_data_dict[full_key] # List of 5 lists of (eval, fitness)
            
            # Extract fitness values for all 5 trials
            all_fitness_values = []
            
            for trial in raw_trials:
                # Extract only the fitness values 
                fitnesses = [f for e, f in trial]
                all_fitness_values.append(fitnesses)
                
            if all_fitness_values:
                # Convert to np array and calculate the mean fitness across the 5 trials
                mean_fitness = np.mean(all_fitness_values, axis=0)
                
                # Plot the mean curve
                plt.plot(fixed_evaluations, mean_fitness, style, label=label, linewidth=2)

    plt.title(f'Figure: {title}', fontsize=14)
    plt.xlabel('Fitness Evaluations', fontsize=12)
    plt.ylabel('Mean Best Fitness Achieved', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close() 


# --- Execution of Experiments ---
if __name__ == "__main__":
    # Defining the 4 parameter settings
    settings = [
        {'pm': 0.01, 't': 3},
        {'pm': 0.05, 't': 3},
        {'pm': 0.01, 't': 7},
        {'pm': 0.05, 't': 7}
    ]

    ALL_RESULTS = {}
    ALL_CONVERGENCE_DATA = {}

    # 1. Run BPP1 Experiments
    bpp1_weights = get_bpp1_weights(K_ITEMS)
    for setting in settings:
        key = f"BPP1_pm{setting['pm']}_t{setting['t']}"
        results, data = run_experiment_set("BPP1", BINS_BPP1, bpp1_weights, P_SIZE, setting['pm'], setting['t'], pc_rate=PC)
        ALL_RESULTS[key] = results
        ALL_CONVERGENCE_DATA[key] = data

    # 2. Run BPP2 Experiments
    bpp2_weights = get_bpp2_weights(K_ITEMS)
    for setting in settings:
        key = f"BPP2_pm{setting['pm']}_t{setting['t']}"
        results, data = run_experiment_set("BPP2", BINS_BPP2, bpp2_weights, P_SIZE, setting['pm'], setting['t'], pc_rate=PC)
        ALL_RESULTS[key] = results
        ALL_CONVERGENCE_DATA[key] = data
        
    print("\n All 40 trials completed. Final Results Dict: ")
    
    # This loop is for gen of tables and graphs
    for key, results in ALL_RESULTS.items():
        ds = [r['final_d'] for r in results]
        mean_d = np.mean(ds)
        std_d = np.std(ds)
        print(f"\n{key}:")
        print(f"  Final d (Mean): {mean_d:.2f}")
        print(f"  Final d (Std Dev): {std_d:.2f}")
        print(f"  Best d found: {min(ds):.2f}")
    
    # BPP1 Graph
    plot_convergence_curves(
        ALL_CONVERGENCE_DATA, 
        'BPP1', 
        'BPP1 (Linear Weights) Convergence - Influence of pm and t', 
        'Figure1_BPP1_Convergence.png'
    )

    # BPP2 Graph
    plot_convergence_curves(
        ALL_CONVERGENCE_DATA, 
        'BPP2', 
        'BPP2 (Quadratic Weights) Convergence - Influence of pm and t', 
        'Figure2_BPP2_Convergence.png'
    )

    print("\nConvergence graphs generated: Figure1_BPP1_Convergence.png and Figure2_BPP2_Convergence.png")

# --- Further Experiments ---
# Experiment 1: Influence of Population Size (p=200)
#    - Goal: Test if a larger search pool (p=200) is more effective than
#            a larger number of generations (p=100, 100 generations).
#    - Hypothesis: P=200 will be worse because the generation count is halved (50 generations).

# Experiment 2: Influence of Crossover Rate (pc=0.5)
#    - Goal: Test the algorithm's heavy reliance on the crossover operator.
#    - Hypothesis: Reducing pc from 0.8 to 0.5 will significantly degrade 
#                  performance, confirming that crossover is the main 
#                  source of beneficial exploration (since mutation is destructive).
    print("\nRunning Further Experiments")

    # Use BPP1 weights for these tests only
    weights = get_bpp1_weights(K_ITEMS)
    num_bins = BINS_BPP1 

    # Experiment 1: Influence of Population Size (p=200) 
    P_EXP1 = 200 # Variable changed
    PM_EXP1 = 0.01 
    T_EXP1 = 7
    PC_EXP1 = 0.8 # Fixed

    key_e1 = f"Further_Exp1_p{P_EXP1}"
    results_e1, _ = run_experiment_set("BPP1 (p=200)", num_bins, weights, P_EXP1, PM_EXP1, T_EXP1, pc_rate=PC_EXP1)
    ALL_RESULTS[key_e1] = results_e1
    
    # Analyse and print 1 results
    ds_e1 = [r['final_d'] for r in results_e1]
    print(f"E1 Results (p=200): Mean d={np.mean(ds_e1):.2f}, Std Dev={np.std(ds_e1):.2f}")


    # Experiment 2: Influence of Crossover Rate (pc=0.5) 
    P_EXP2 = 100 
    PM_EXP2 = 0.01 
    T_EXP2 = 7
    PC_EXP2 = 0.5 # Variable changed

    key_e2 = f"Further_Exp2_pc{PC_EXP2}"
    results_e2, _ = run_experiment_set("BPP1 (pc=0.5)", num_bins, weights, P_EXP2, PM_EXP2, T_EXP2, pc_rate=PC_EXP2)
    ALL_RESULTS[key_e2] = results_e2
    
    # Analyse and print results
    ds_e2 = [r['final_d'] for r in results_e2]
    print(f"E2 Results (pc=0.5): Mean d={np.mean(ds_e2):.2f}, Std Dev={np.std(ds_e2):.2f}")
    
    print("All Further Experiments Completed ")
    print("End of all experiments, Thank you :)")