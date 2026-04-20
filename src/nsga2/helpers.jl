using EvoLP

# Project 3 framing: feature selection is a tradeoff between model performance
# and number of selected features.
function nsga2_objectives(individual::AbstractVector{Bool}, landscape::Landscape)
    count(individual) == 0 && return (-Inf, -Inf)
    return (fitness(individual, landscape; penalized=false), -Float64(count(individual)))
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
