function run_nsga2(landscape::Landscape;
                   pop_size::Int=100,
                   generations::Int=500,
                   mutation_rate::Float64=0.01)
    population = deduplicate_and_refill(
        binary_vector_pop(pop_size, landscape.n_features),
        pop_size,
        landscape.n_features,
    )
    front_size_history = Int[]
    entropy_history = Float64[]

    for _ in 1:generations
        objectives = [nsga2_objectives(ind, landscape) for ind in population]
        push!(front_size_history, length(first(non_dominated_fronts(objectives))))
        push!(entropy_history, population_entropy(population))

        ranks, crowding = rank_and_crowding(objectives)
        offspring = make_offspring(
            population;
            pop_size=pop_size,
            mutation_rate=mutation_rate,
            ranks=ranks,
            crowding=crowding,
        )

        combined_population = deduplicate_and_refill(
            vcat(population, offspring),
            pop_size,
            landscape.n_features,
        )
        combined_objectives = [nsga2_objectives(ind, landscape) for ind in combined_population]
        population = select_survivors(combined_population, combined_objectives, pop_size)
    end

    objectives = [nsga2_objectives(ind, landscape) for ind in population]
    pareto_front = first(non_dominated_fronts(objectives))
    pareto_population, _ = deduplicate_individuals(population[pareto_front])
    pareto_objectives = [nsga2_objectives(ind, landscape) for ind in pareto_population]

    return pareto_population,
           pareto_objectives,
           front_size_history,
           entropy_history
end
