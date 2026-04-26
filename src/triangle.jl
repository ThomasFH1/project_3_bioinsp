struct WeightLookupLandscape
    fitness_by_weight::Vector{UInt8}
    n_features::Int
    name::String
end

function triangle_fitness_from_weight(n_active::Int; m::Int=1, s::Int=4)
    if s <= 0
        error("Step size s must be positive.")
    end

    period = 2 * s
    position = n_active % period
    distance_to_peak = min(position, period - position)
    return m * distance_to_peak
end

function triangle_fitness(x::AbstractVector{Bool}; n::Int=16, m::Int=1, s::Int=4)
    if length(x) != n
        error("Expected bitstring of length $n, got $(length(x)).")
    end

    return triangle_fitness_from_weight(count(x); m=m, s=s)
end

function triangle_weight_lookup(; n::Int=16, m::Int=1, s::Int=4)
    return UInt8[triangle_fitness_from_weight(k; m=m, s=s) for k in 0:n]
end

function test_triangle_weight_lookup()
    return UInt8[
        0, 1, 2, 3, 4, 5, 4, 3,
        2, 1, 0, 1, 2, 3, 4, 5,
        4, 3, 2, 1, 0, 1, 2, 3,
        4, 5, 4, 3, 2, 1, 0, 6,
    ]
end

function triangle_landscape(; n::Int=16, m::Int=1, s::Int=4, name::String="triangle")
    return WeightLookupLandscape(triangle_weight_lookup(n=n, m=m, s=s), n, name)
end

function test_triangle_landscape(; name::String="test_triangle")
    return WeightLookupLandscape(test_triangle_weight_lookup(), 31, name)
end

function fitness(individual::AbstractVector{Bool}, landscape::WeightLookupLandscape; penalized::Bool=true)
    if length(individual) != landscape.n_features
        error("Expected bitstring of length $(landscape.n_features), got $(length(individual)).")
    end

    return Float64(landscape.fitness_by_weight[count(individual) + 1])
end

function accuracy(individual::AbstractVector{Bool}, landscape::WeightLookupLandscape)
    return fitness(individual, landscape)
end
