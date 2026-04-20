function run_nsga2(landscape::Landscape;
                   pop_size::Int=100,
                   generations::Int=500,
                   mutation_rate::Float64=0.01)
    population = binary_vector_pop(pop_size, landscape.n_features)
    front_size_history = Int[]
    entropy_history = Float64[]

    for _ in 1:generations
        objectives = [nsga2_objectives(ind, landscape) for ind in population]
        push!(front_size_history, length(pareto_front_indices(objectives)))
        push!(entropy_history, population_entropy(population))

        offspring = make_offspring(population; pop_size, mutation_rate)
        combined_population = vcat(population, offspring)
        combined_objectives = [nsga2_objectives(ind, landscape) for ind in combined_population]
        population = select_survivors(combined_population, combined_objectives, pop_size)
    end

    objectives = [nsga2_objectives(ind, landscape) for ind in population]
    pareto_front = pareto_front_indices(objectives)

    return copy.(population[pareto_front]),
           objectives[pareto_front],
           front_size_history,
           entropy_history
end
