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

function pareto_front_indices(objectives)
    front = Int[]

    for candidate in eachindex(objectives)
        is_dominated = false

        for other in eachindex(objectives)
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

    return front
end

function crowding_scores(front, objectives)
    # TODO: replace with standard NSGA-II crowding distance over both objectives.
    return Dict(i => 0.0 for i in front)
end

function make_offspring(population; pop_size::Int, mutation_rate::Float64)
    recombinator = EvoLP.TwoPointRecombinator()
    mutator = EvoLP.BitwiseMutator(mutation_rate)
    offspring = BitVector[]

    while length(offspring) < pop_size
        p1, p2 = rand(1:length(population), 2)
        child = EvoLP.cross(recombinator, population[p1], population[p2])
        push!(offspring, mutate(mutator, child))
    end

    return offspring
end

function select_survivors(population, objectives, pop_size::Int)
    # TODO: full NSGA-II should add fronts in rank order, then use crowding
    # distance only when the next front does not fit.
    front = pareto_front_indices(objectives)
    crowding = crowding_scores(front, objectives)
    selected = sort(front; by=i -> crowding[i], rev=true)

    if length(selected) < pop_size
        remaining = setdiff(collect(eachindex(population)), selected)
        append!(selected, remaining[1:(pop_size - length(selected))])
    end

    return copy.(population[selected[1:pop_size]])
end
