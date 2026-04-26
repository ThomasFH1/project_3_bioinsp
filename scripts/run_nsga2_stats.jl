using Pkg

project_dir = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(project_dir)

using Random
using Statistics
using Project3Bioinsp

function parse_arg(args::Vector{String}, index::Int, default, parser)
    if length(args) < index
        return default
    end

    return parser(args[index])
end

function load_stats_landscape(project_dir::String, source::String, epsilon::Float64)
    normalized = lowercase(replace(source, "-" => "_"))

    if normalized in ("triangle", "train_triangle", "training_triangle")
        return triangle_landscape(; n=16, m=1, s=4, name="train_triangle")
    end

    if normalized in ("test_triangle", "triangle_test")
        return test_triangle_landscape()
    end

    dataset_path = isabspath(source) ? source : normpath(joinpath(project_dir, source))
    landscape_name = splitext(basename(dataset_path))[1]
    return load_landscape(dataset_path; ε=epsilon, name=landscape_name)
end

function unique_pair_count(pareto, objectives)
    pairs = Set{Tuple{Float64,Int}}()

    for (individual, objective) in zip(pareto, objectives)
        push!(pairs, (objective[1], count(individual)))
    end

    return length(pairs)
end

function run_landscape_stats(
    landscape;
    n_seeds::Int,
    generations::Int,
    pop_size::Int,
    mutation_rate::Float64,
)
    counts = Int[]

    println("Dataset: $(landscape.name)")
    for seed in 1:n_seeds
        Random.seed!(seed)
        pareto, objectives, _, _ = run_nsga2(
            landscape;
            pop_size=pop_size,
            generations=generations,
            mutation_rate=mutation_rate,
            trace=false,
        )

        n_solutions = unique_pair_count(pareto, objectives)
        push!(counts, n_solutions)
        println("  seed=$(lpad(seed, 2)) n_solutions=$(n_solutions)")
    end

    println("  counts=$(counts)")
    println("  mean=$(round(mean(counts); digits=4))")
    println("  sample_std=$(round(std(counts); digits=4))")
    println("  sample_variance=$(round(var(counts); digits=4))")
    println()
end

function main(args::Vector{String})
    n_seeds = parse_arg(args, 1, 10, value -> parse(Int, value))
    generations = parse_arg(args, 2, 500, value -> parse(Int, value))
    pop_size = parse_arg(args, 3, 100, value -> parse(Int, value))
    mutation_rate = parse_arg(args, 4, 0.01, value -> parse(Float64, value))
    epsilon = parse_arg(args, 5, 0.1, value -> parse(Float64, value))
    dataset_args = length(args) >= 6 ? args[6:end] : [
        "data/05-credit-a_rf_F.h5",
        "data/08-letter-r_knn_F.h5",
    ]

    println("NSGA-II Pareto pair-count stats")
    println("n_seeds=$(n_seeds), generations=$(generations), pop_size=$(pop_size), mutation_rate=$(mutation_rate), epsilon=$(epsilon)")
    println()

    for dataset_arg in dataset_args
        landscape = load_stats_landscape(project_dir, dataset_arg, epsilon)
        run_landscape_stats(
            landscape;
            n_seeds=n_seeds,
            generations=generations,
            pop_size=pop_size,
            mutation_rate=mutation_rate,
        )
    end
end

main(ARGS)
