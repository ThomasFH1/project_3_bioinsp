function run_nsga2(landscape;
                   pop_size::Int=100,
                   generations::Int=500,
                   mutation_rate::Float64=0.01,
                   trace::Bool=false,
                   trace_interval::Int=50,
                   trace_limit::Int=12,
                   trace_verbose::Bool=false,
                   trace_io::IO=stdout)
    if trace_interval <= 0
        error("trace_interval must be positive.")
    end

    population = binary_vector_pop(pop_size, landscape.n_features)
    front_size_history = Int[]
    entropy_history = Float64[]

    if trace
        objectives = [nsga2_objectives(ind, landscape) for ind in population]
        fronts = non_dominated_fronts(objectives)
        print_nsga2_trace(
            trace_io,
            0,
            population,
            objectives,
            fronts,
            population_entropy(population);
            limit=trace_limit,
            verbose=trace_verbose,
        )
    end

    for generation in 1:generations
        objectives = [nsga2_objectives(ind, landscape) for ind in population]
        fronts = non_dominated_fronts(objectives)
        push!(front_size_history, length(first(fronts)))
        push!(entropy_history, population_entropy(population))

        ranks, crowding = rank_and_crowding(objectives)
        offspring = make_offspring(
            population;
            pop_size=pop_size,
            mutation_rate=mutation_rate,
            ranks=ranks,
            crowding=crowding,
        )

        combined_population = vcat(population, offspring)
        combined_objectives = [nsga2_objectives(ind, landscape) for ind in combined_population]
        population = select_survivors(combined_population, combined_objectives, pop_size)

        if trace && (generation == 1 || generation % trace_interval == 0 || generation == generations)
            trace_objectives = [nsga2_objectives(ind, landscape) for ind in population]
            trace_fronts = non_dominated_fronts(trace_objectives)
            print_nsga2_trace(
                trace_io,
                generation,
                population,
                trace_objectives,
                trace_fronts,
                population_entropy(population);
                limit=trace_limit,
                verbose=trace_verbose,
            )
        end
    end

    objectives = [nsga2_objectives(ind, landscape) for ind in population]
    pareto_front = first(non_dominated_fronts(objectives))

    return copy.(population[pareto_front]),
           objectives[pareto_front],
           front_size_history,
           entropy_history
end
