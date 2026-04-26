using HDF5
using Statistics

struct Landscape
    values::Vector{Float64}
    n_features::Int
    name::String
    penalty_weight::Float64
end

function infer_n_features(n_rows::Int)
    n_features = round(Int, log2(n_rows + 1))
    if 2^n_features - 1 != n_rows
        error("Landscape row count $n_rows is not of the form 2^n - 1.")
    end
    return n_features
end

function bitstring_to_index(individual::AbstractVector{Bool})
    return sum(individual[i] * 2^(i - 1) for i in eachindex(individual))
end

function fitness(individual::AbstractVector{Bool}, landscape::Landscape; penalized::Bool=true)
    if length(individual) != landscape.n_features
        error("Expected bitstring of length $(landscape.n_features), got $(length(individual)).")
    end

    index = bitstring_to_index(individual)
    if index == 0
        return -Inf
    end

    value = landscape.values[index]
    if penalized
        return apply_penalty(value, count(individual), landscape.penalty_weight)
    end

    return value
end

function accuracy(individual::AbstractVector{Bool}, landscape::Landscape)
    return fitness(individual, landscape; penalized=false)
end

function population_entropy(pop)
    n_individuals = length(pop)
    n_bits = length(pop[1])
    H_entropy = 0.0
    for i in 1:n_bits
        p = sum(individual[i] for individual in pop) / n_individuals
        if p > 0 && p < 1
            H_entropy -= p * log2(p) + (1 - p) * log2(1 - p)
        end
    end
    return H_entropy / n_bits
end

function load_landscape(filepath::String; ε::Float64=0.1, name::String=splitext(basename(filepath))[1])
    accuracy_samples = h5open(filepath, "r") do file
        read(file["accuracies"])
    end

    mean_accuracies = vec(Float64.(mean(accuracy_samples, dims=2)))
    n_rows = length(mean_accuracies)
    n_features = infer_n_features(n_rows)

    return Landscape(mean_accuracies, n_features, name, ε)
end

function apply_penalty(accuracy::Float64, n_features::Int, ε::Float64)
    return accuracy - ε * n_features
end

function index_to_bitstring(index::Int, n_features::Int)
    bits = falses(n_features)
    for i in 1:n_features
        bits[i] = ((index >> (i - 1)) & 1) == 1
    end
    return bits
end

function local_optima_mask(landscape::Landscape; strict::Bool=true)
    n_values = length(landscape.values)
    mask = falses(n_values)

    for index in 1:n_values
        bits = index_to_bitstring(index, landscape.n_features)
        current_fitness = fitness(bits, landscape)
        is_optimum = true

        for bit in 1:landscape.n_features
            neighbor = copy(bits)
            neighbor[bit] = !neighbor[bit]
            neighbor_fitness = fitness(neighbor, landscape)

            if strict ? neighbor_fitness >= current_fitness : neighbor_fitness > current_fitness
                is_optimum = false
                break
            end
        end

        mask[index] = is_optimum
    end

    return mask
end
