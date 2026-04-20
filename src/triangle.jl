function triangle_fitness(x::BitVector; n::Int=16, m::Int=1, s::Int=4)
    if length(x) != n
        error("Expected bitstring of length $n, got $(length(x)).")
    end

    if s <= 0
        error("Step size s must be positive.")
    end

    n_active = count(x)
    period = 2 * s
    position = n_active % period

    distance_to_peak = min(position, period - position)
    return m * distance_to_peak
end
