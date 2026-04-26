
using Statistics
 
# Sigmoid helper function. Converts any number into a value between 0 and 1.
# We use it to turn velocity into a probability for bit flipping.
sigmoid(v) = 1.0 / (1.0 + exp(-v))
 
function run_pso(landscape;
                 swarm_size::Int=50,
                 iterations::Int=500,
                 w::Float64=0.7,        # inertia: how much old velocity is kept
                 c1::Float64=1.5,       # cognitive: pull toward personal best
                 c2::Float64=1.5,       # social: pull toward global best
                 v_max::Float64=4.0,    # clamp velocity so sigmoid stays meaningful
                 trace::Bool=false)
 
    n = landscape.n_features
 
    # Initialise positions and velocities
    positions  = [BitVector(rand(Bool, n)) for _ in 1:swarm_size]
    velocities = [rand(Float64, n) .* 2 .- 1 for _ in 1:swarm_size]  # uniform in [-1, 1]
 
    # At the start, every particle's personal best is just its starting position.
    personal_best_pos = copy.(positions)
    personal_best_fit = [fitness(p, landscape) for p in positions]
 
    # The global best is whichever particle started with the highest fitness.
    gbest_idx = argmax(personal_best_fit)
    global_best_pos = copy(personal_best_pos[gbest_idx])
    global_best_fit = personal_best_fit[gbest_idx]
 
    # Tracking
    max_history     = Float64[]
    mean_history    = Float64[]
    min_history     = Float64[]
    entropy_history = Float64[]
    trace_history   = NamedTuple[]
 
    for iter in 1:iterations
        fits = [fitness(positions[i], landscape) for i in 1:swarm_size]
 
        push!(max_history,     maximum(fits))
        push!(mean_history,    mean(fits))
        push!(min_history,     minimum(fits))
        push!(entropy_history, population_entropy(positions))
 
        for i in 1:swarm_size
            r1 = rand(Float64, n)
            r2 = rand(Float64, n)
 
            # Velocity update: pull toward personal best and global best with randomness
            velocities[i] = w .* velocities[i] .+
                             c1 .* r1 .* (Float64.(personal_best_pos[i]) .- Float64.(positions[i])) .+
                             c2 .* r2 .* (Float64.(global_best_pos)      .- Float64.(positions[i]))
 
            # Clamp velocity so sigmoid doesn't saturate
            velocities[i] = clamp.(velocities[i], -v_max, v_max)
 
            # Update position: each bit is 1 with probability sigmoid(velocity)
            for j in 1:n
                positions[i][j] = rand() < sigmoid(velocities[i][j])
            end
 
            # Update personal best
            f = fitness(positions[i], landscape)
            if f > personal_best_fit[i]
                personal_best_fit[i] = f
                personal_best_pos[i] = copy(positions[i])
            end
 
            # Update global best
            if personal_best_fit[i] > global_best_fit
                global_best_fit = personal_best_fit[i]
                global_best_pos = copy(personal_best_pos[i])
            end
        end

        if trace
            push!(
                trace_history,
                (
                    iteration=iter,
                    global_best=copy(global_best_pos),
                    global_best_fitness=global_best_fit,
                ),
            )
        end
    end

    if trace
        return global_best_pos, global_best_fit, max_history, mean_history, min_history, entropy_history, trace_history
    end

    return global_best_pos, global_best_fit, max_history, mean_history, min_history, entropy_history
end
