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

function summarize_front(pareto, objectives)
    by_weight = Dict{Int,Float64}()

    for (individual, objective) in zip(pareto, objectives)
        weight = count(individual)
        by_weight[weight] = max(get(by_weight, weight, -Inf), objective[1])
    end

    println()
    println("Final Pareto front by active-bit count:")
    for weight in sort(collect(keys(by_weight)))
        println("  weight=$(lpad(weight, 2)) fitness=$(by_weight[weight])")
    end
end

function main(args::Vector{String})
    seed = parse_arg(args, 1, 1, value -> parse(Int, value))
    generations = parse_arg(args, 2, 500, value -> parse(Int, value))
    trace_interval = parse_arg(args, 3, 50, value -> parse(Int, value))
    trace_verbose = parse_arg(args, 4, false, value -> parse(Bool, value))

    Random.seed!(seed)

    landscape = test_triangle_landscape()
    println("Running NSGA-II on $(landscape.name)")
    println("seed=$(seed), generations=$(generations), trace_interval=$(trace_interval), trace_verbose=$(trace_verbose)")

    pareto, objectives, _, _ = run_nsga2(
        landscape;
        pop_size=100,
        generations=generations,
        mutation_rate=0.01,
        trace=true,
        trace_interval=trace_interval,
        trace_verbose=trace_verbose,
    )

    summarize_front(pareto, objectives)

    best_index = argmax(first.(objectives))
    best_individual = pareto[best_index]
    best_objective = objectives[best_index]

    println()
    println("Best fitness found: $(best_objective[1])")
    println("Best active bits: $(count(best_individual))")
    println("All-31 optimum reached: $(count(best_individual) == 31 && best_objective[1] == 6.0)")
end

main(ARGS)
