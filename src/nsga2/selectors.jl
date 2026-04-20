function rank_and_crowding(objectives)
    ranks = zeros(Int, length(objectives))
    crowding = zeros(Float64, length(objectives))

    for (rank, front) in enumerate(non_dominated_fronts(objectives))
        scores = crowding_scores(front, objectives)

        for index in front
            ranks[index] = rank
            crowding[index] = scores[index]
        end
    end

    return ranks, crowding
end

function crowded_better(a::Int, b::Int, ranks, crowding)
    if ranks[a] != ranks[b]
        return ranks[a] < ranks[b]
    end

    if crowding[a] != crowding[b]
        return crowding[a] > crowding[b]
    end

    return rand(Bool)
end

function crowded_tournament(ranks, crowding)
    a = rand(1:length(ranks))
    b = rand(1:length(ranks))
    return crowded_better(a, b, ranks, crowding) ? a : b
end

function select_survivors(population, objectives, pop_size::Int)
    selected = Int[]

    for front in non_dominated_fronts(objectives)
        if length(selected) + length(front) <= pop_size
            append!(selected, front)
        else
            crowding = crowding_scores(front, objectives)
            sorted_front = sort(front; by=i -> crowding[i], rev=true)
            remaining_slots = pop_size - length(selected)
            append!(selected, sorted_front[1:remaining_slots])
            break
        end

        if length(selected) == pop_size
            break
        end
    end

    return copy.(population[selected])
end
