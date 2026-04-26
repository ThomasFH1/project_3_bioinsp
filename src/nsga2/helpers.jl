using EvoLP

function nsga2_penalty_weight(landscape)
    if hasproperty(landscape, :penalty_weight)
        return Float64(getproperty(landscape, :penalty_weight))
    end

    return 0.0
end

# Project 3 framing: feature selection is a tradeoff between model performance
# and the regularization penalty. For synthetic landscapes without a penalty,
# the second objective is constant and NSGA-II behaves as a single-objective run.
function nsga2_objectives(individual::AbstractVector{Bool}, landscape)
    count(individual) == 0 && return (-Inf, -Inf)
    penalty = nsga2_penalty_weight(landscape) * count(individual)
    return (fitness(individual, landscape; penalized=false), -penalty)
end

function dominates(a, b)
    return all(a[i] >= b[i] for i in eachindex(a)) &&
           any(a[i] > b[i] for i in eachindex(a))
end

function non_dominated_fronts(objectives)
    remaining = collect(eachindex(objectives))
    fronts = Vector{Vector{Int}}()

    while !isempty(remaining)
        front = Int[]

        for candidate in remaining
            is_dominated = false

            for other in remaining
                if other == candidate
                    continue
                end

                if dominates(objectives[other], objectives[candidate])
                    is_dominated = true
                    break
                end
            end

            if !is_dominated
                push!(front, candidate)
            end
        end

        push!(fronts, front)
        remaining = setdiff(remaining, front)
    end

    return fronts
end

function crowding_scores(front, objectives)
    scores = Dict(i => 0.0 for i in front)

    if length(front) <= 2
        for i in front
            scores[i] = Inf
        end
        return scores
    end

    n_objectives = length(objectives[first(front)])

    for objective_index in 1:n_objectives
        sorted_front = sort(front; by=i -> objectives[i][objective_index])

        scores[first(sorted_front)] = Inf
        scores[last(sorted_front)] = Inf

        min_value = objectives[first(sorted_front)][objective_index]
        max_value = objectives[last(sorted_front)][objective_index]
        value_range = max_value - min_value

        if value_range == 0 || !isfinite(value_range)
            continue
        end

        for position in 2:(length(sorted_front) - 1)
            previous = sorted_front[position - 1]
            current = sorted_front[position]
            next = sorted_front[position + 1]

            scores[current] +=
                (objectives[next][objective_index] -
                 objectives[previous][objective_index]) / value_range
        end
    end

    return scores
end

function make_offspring(population; pop_size::Int, mutation_rate::Float64, ranks, crowding)
    recombinator = EvoLP.TwoPointRecombinator()
    mutator = EvoLP.BitwiseMutator(mutation_rate)
    offspring = BitVector[]

    while length(offspring) < pop_size
        p1 = crowded_tournament(ranks, crowding)
        p2 = crowded_tournament(ranks, crowding)
        child = EvoLP.cross(recombinator, population[p1], population[p2])
        push!(offspring, mutate(mutator, child))
    end

    return offspring
end

function format_trace_number(value)
    if value isa Real && isfinite(value)
        return string(round(value; digits=4))
    end

    return string(value)
end

function format_weight_counts(population; limit::Int=12)
    counts = Dict{Int,Int}()

    for individual in population
        weight = count(individual)
        counts[weight] = get(counts, weight, 0) + 1
    end

    pairs = sort(collect(counts); by=first)
    shown = first(pairs, min(limit, length(pairs)))
    text = join(["$weight:$n" for (weight, n) in shown], " ")

    if length(pairs) > limit
        text *= " ..."
    end

    return text
end

function format_front_weights(front, population, objectives; limit::Int=12)
    best_by_weight = Dict{Int,Float64}()

    for index in front
        weight = count(population[index])
        fitness_value = objectives[index][1]
        best_by_weight[weight] = max(get(best_by_weight, weight, -Inf), fitness_value)
    end

    pairs = sort(collect(best_by_weight); by=first)
    shown = first(pairs, min(limit, length(pairs)))
    text = join(["$weight=>$(format_trace_number(fitness_value))" for (weight, fitness_value) in shown], " ")

    if length(pairs) > limit
        text *= " ..."
    end

    return text
end

function sampled_front_points(front, population, objectives)
    ordered_front = sort(front; by=i -> (count(population[i]), -objectives[i][1]))
    sample_positions = Int[]

    for position in eachindex(ordered_front)
        if position <= 10 || position % 10 == 0
            push!(sample_positions, position)
        end
    end

    return [
        (
            position,
            count(population[ordered_front[position]]),
            objectives[ordered_front[position]][1],
            objectives[ordered_front[position]][2],
        )
        for position in sample_positions
    ]
end

function format_front_point(row)
    position, weight, fitness_value, penalty_objective = row
    return "#$(position):bits=$(weight),fit=$(format_trace_number(fitness_value)),penalty_obj=$(format_trace_number(penalty_objective))"
end

function print_nsga2_trace(
    io::IO,
    generation::Int,
    population,
    objectives,
    fronts,
    entropy::Float64;
    limit::Int=12,
    verbose::Bool=false,
)
    first_front = first(fronts)
    fitness_values = first.(objectives)
    best_index = argmax(fitness_values)
    best_fitness = objectives[best_index][1]
    best_weight = count(population[best_index])

    println(
        io,
        "gen=$(lpad(generation, 4)) ",
        "front=$(lpad(length(first_front), 3)) ",
        "best_fit=$(lpad(format_trace_number(best_fitness), 7)) ",
        "best_bits=$(lpad(best_weight, 2)) ",
        "entropy=$(format_trace_number(entropy))",
    )
    println(io, "  front weights fitness: ", format_front_weights(first_front, population, objectives; limit=limit))
    if verbose
        println(io, "  sampled frontier:      ", join(format_front_point.(sampled_front_points(first_front, population, objectives)), " | "))
    end
    println(io, "  population weights:     ", format_weight_counts(population; limit=limit))
end
