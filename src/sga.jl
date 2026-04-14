using EvoLP
using Statistics

function run_sga(landscape::Landscape;
                 pop_size=100, generations=500,
                 mutation_rate=0.01, tournament_size=3,
                 num_elite=nothing)
    n_features = landscape.n_features

    if num_elite === nothing
        num_elite = div(pop_size, 5)
    end

    population = binary_vector_pop(pop_size, n_features)

    selector = EvoLP.TournamentSelector(tournament_size)
    recombinator = EvoLP.TwoPointRecombinator()
    mutator = EvoLP.BitwiseMutator(mutation_rate)

    best_fitness = -Inf
    best_individual = nothing
    max_history = Float64[]
    mean_history = Float64[]
    min_history = Float64[]
    entropy_history = Float64[]

    for gen in 1:generations
        population_fitness = [fitness(ind, landscape) for ind in population]

        push!(max_history, maximum(population_fitness))
        push!(mean_history, mean(population_fitness))
        push!(min_history, minimum(population_fitness))
        push!(entropy_history, population_entropy(population))

        gen_best = argmax(population_fitness)
        if population_fitness[gen_best] > best_fitness
            best_fitness = population_fitness[gen_best]
            best_individual = copy(population[gen_best])
        end

        best_indices = sortperm(population_fitness, rev=true)
        new_population = BitVector[]

        for i in 1:num_elite
            push!(new_population, population[best_indices[i]])
        end

        while length(new_population) < pop_size
            parent_positions = EvoLP.select(selector, -population_fitness)
            child1 = EvoLP.cross(recombinator, population[parent_positions[1]], population[parent_positions[2]])
            child2 = EvoLP.cross(recombinator, population[parent_positions[2]], population[parent_positions[1]])
            push!(new_population, mutate(mutator, child1))
            if length(new_population) < pop_size
                push!(new_population, mutate(mutator, child2))
            end
        end

        population = new_population
    end

    return best_individual, best_fitness, max_history, mean_history, min_history, entropy_history
end
