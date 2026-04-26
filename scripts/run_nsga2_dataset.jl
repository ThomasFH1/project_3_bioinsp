using Pkg

project_dir = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(project_dir)

using Random
using Project3Bioinsp

function parse_arg(args::Vector{String}, index::Int, default, parser)
    if length(args) < index
        return default
    end

    return parser(args[index])
end

function resolve_dataset_path(project_dir::String, path::String)
    return isabspath(path) ? path : normpath(joinpath(project_dir, path))
end

function unique_front_rows(pareto, objectives)
    seen = Set{Tuple{Float64,Int}}()
    rows = []

    for (individual, objective) in zip(pareto, objectives)
        accuracy_value = objective[1]
        selected = count(individual)
        key = (accuracy_value, selected)

        if !(key in seen)
            push!(seen, key)
            push!(rows, (accuracy_value, selected, objective[2], individual))
        end
    end

    sort!(rows; by=row -> (row[2], -row[1]))
    return rows
end

function print_front_rows(rows; limit::Int=20)
    println("Pareto front sample: accuracy, selected_features, second_objective")

    for row in rows[1:min(limit, length(rows))]
        accuracy_value, selected, second_objective, _ = row
        println(
            "  ",
            round(accuracy_value; digits=6),
            "  ",
            selected,
            "  ",
            round(second_objective; digits=6),
        )
    end

    if length(rows) > limit
        println("  ... $(length(rows) - limit) more")
    end
end

function best_penalized_solution(pareto, landscape)
    best_value = -Inf
    best_individual = nothing

    for individual in pareto
        value = accuracy(individual, landscape) - landscape.penalty_weight * count(individual)
        if value > best_value
            best_value = value
            best_individual = individual
        end
    end

    return best_value, best_individual
end

function main(args::Vector{String})
    if isempty(args)
        error("Usage: run_nsga2_dataset.jl DATASET_PATH [seed] [generations] [pop_size] [mutation_rate] [epsilon] [trace] [trace_interval] [trace_verbose]")
    end

    dataset_path = resolve_dataset_path(project_dir, args[1])
    seed = parse_arg(args, 2, 1, value -> parse(Int, value))
    generations = parse_arg(args, 3, 500, value -> parse(Int, value))
    pop_size = parse_arg(args, 4, 100, value -> parse(Int, value))
    mutation_rate = parse_arg(args, 5, 0.01, value -> parse(Float64, value))
    epsilon = parse_arg(args, 6, 0.1, value -> parse(Float64, value))
    trace = parse_arg(args, 7, false, value -> parse(Bool, value))
    trace_interval = parse_arg(args, 8, 50, value -> parse(Int, value))
    trace_verbose = parse_arg(args, 9, false, value -> parse(Bool, value))

    Random.seed!(seed)

    landscape_name = splitext(basename(dataset_path))[1]
    landscape = load_landscape(dataset_path; ε=epsilon, name=landscape_name)

    println("Running NSGA-II on $(landscape.name)")
    println("seed=$(seed), generations=$(generations), pop_size=$(pop_size), mutation_rate=$(mutation_rate), epsilon=$(epsilon)")

    pareto, objectives, front_size_history, entropy_history = run_nsga2(
        landscape;
        pop_size=pop_size,
        generations=generations,
        mutation_rate=mutation_rate,
        trace=trace,
        trace_interval=trace_interval,
        trace_verbose=trace_verbose,
    )

    rows = unique_front_rows(pareto, objectives)
    best_accuracy_index = argmax(first.(objectives))
    best_accuracy_individual = pareto[best_accuracy_index]
    best_accuracy_objective = objectives[best_accuracy_index]
    best_penalized, best_penalized_individual = best_penalized_solution(pareto, landscape)

    println()
    println("Dataset: $(landscape.name)")
    println("Features: $(landscape.n_features)")
    println("Pareto individuals: $(length(pareto))")
    println("Unique Pareto objective/count pairs: $(length(rows))")
    println("Final front size history value: $(last(front_size_history))")
    println("Final entropy: $(round(last(entropy_history); digits=4))")
    println()
    print_front_rows(rows)
    println()
    println("Best raw accuracy: $(round(best_accuracy_objective[1]; digits=6)) with $(count(best_accuracy_individual)) features")
    println("Best penalized fitness on Pareto front: $(round(best_penalized; digits=6)) with $(count(best_penalized_individual)) features and accuracy $(round(accuracy(best_penalized_individual, landscape); digits=6))")
end

main(ARGS)
