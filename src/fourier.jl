using Statistics

function is_power_of_two(n::Int)
    return n > 0 && (n & (n - 1)) == 0
end

function walsh_hadamard!(values::Vector{Float64})
    n = length(values)
    if !is_power_of_two(n)
        error("Walsh-Hadamard input length must be a power of two, got $n.")
    end

    width = 1
    while width < n
        block = 2 * width
        for start in 1:block:n
            @inbounds for offset in 0:(width - 1)
                left = start + offset
                right = left + width
                a = values[left]
                b = values[right]
                values[left] = a + b
                values[right] = a - b
            end
        end
        width = block
    end

    return values
end

function full_landscape_values(
    landscape::Landscape;
    penalized::Bool=false,
    empty_value=nothing,
)
    n = landscape.n_features
    expected_values = 2^n - 1
    if length(landscape.values) != expected_values
        error("Expected $(expected_values) landscape values for $(n) features, got $(length(landscape.values)).")
    end

    values = Vector{Float64}(undef, 2^n)
    nonempty_values = Vector{Float64}(undef, expected_values)

    for index in 1:expected_values
        value = landscape.values[index]
        if penalized
            value = apply_penalty(value, count_ones(index), landscape.penalty_weight)
        end

        values[index + 1] = value
        nonempty_values[index] = value
    end

    values[1] = empty_value === nothing ? mean(nonempty_values) : Float64(empty_value)
    return values
end

function fourier_coefficients(
    landscape::Landscape;
    penalized::Bool=false,
    empty_value=nothing,
)
    coefficients = full_landscape_values(
        landscape;
        penalized=penalized,
        empty_value=empty_value,
    )
    walsh_hadamard!(coefficients)
    coefficients ./= length(coefficients)
    return coefficients
end

function spectral_energy_by_order(coefficients::AbstractVector{<:Real})
    n_coefficients = length(coefficients)
    if !is_power_of_two(n_coefficients)
        error("Expected a power-of-two number of coefficients, got $n_coefficients.")
    end

    n_features = round(Int, log2(n_coefficients))
    energy = zeros(Float64, n_features + 1)

    for mask in 0:(n_coefficients - 1)
        order = count_ones(mask)
        energy[order + 1] += Float64(coefficients[mask + 1])^2
    end

    return energy
end

function feature_main_effects(coefficients::AbstractVector{<:Real})
    n_coefficients = length(coefficients)
    if !is_power_of_two(n_coefficients)
        error("Expected a power-of-two number of coefficients, got $n_coefficients.")
    end

    n_features = round(Int, log2(n_coefficients))
    return [-2 * Float64(coefficients[2^(feature - 1) + 1]) for feature in 1:n_features]
end

function pair_interaction_effects(coefficients::AbstractVector{<:Real})
    n_coefficients = length(coefficients)
    if !is_power_of_two(n_coefficients)
        error("Expected a power-of-two number of coefficients, got $n_coefficients.")
    end

    n_features = round(Int, log2(n_coefficients))
    interactions = Tuple{Int,Int,Float64,Float64}[]

    for left in 1:(n_features - 1)
        for right in (left + 1):n_features
            mask = 2^(left - 1) + 2^(right - 1)
            coefficient = Float64(coefficients[mask + 1])
            push!(interactions, (left, right, coefficient, 4 * coefficient))
        end
    end

    return interactions
end

function mask_features(mask::Integer, n_features::Int)
    return [feature for feature in 1:n_features if (mask & (1 << (feature - 1))) != 0]
end
