using Pkg

project_dir = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(project_dir)

using Project3Bioinsp

function resolve_dataset_path(project_dir::String, args::Vector{String})
    if isempty(args)
        return joinpath(project_dir, "data", "01-breast-w_lr_F.h5")
    end

    dataset_path = args[1]
    return isabspath(dataset_path) ? dataset_path : normpath(joinpath(project_dir, dataset_path))
end

function print_pareto_summary(pareto_individuals, pareto_objectives; limit::Int=10)
    seen = Set{String}()
    unique_indices = Int[]

    for i in eachindex(pareto_individuals)
        key = "$(pareto_objectives[i]) $(pareto_individuals[i])"
        if !(key in seen)
            push!(seen, key)
            push!(unique_indices, i)
        end
    end

    println("Pareto front size: $(length(pareto_individuals))")
    println("Unique Pareto solutions: $(length(unique_indices))")
    println("Pareto objective sample: accuracy, -selected_features")

    for i in unique_indices[1:min(limit, length(unique_indices))]
        println("  $(pareto_objectives[i])  $(pareto_individuals[i])")
    end

    if length(unique_indices) > limit
        println("  ... $(length(unique_indices) - limit) more")
    end
end

function main(args::Vector{String})
    dataset_path = resolve_dataset_path(project_dir, args)
    landscape_name = splitext(basename(dataset_path))[1]
    landscape = load_landscape(dataset_path; ε=0.1, name=landscape_name)

    best_individual, best_fitness, max_history, mean_history, min_history, entropy_history =
        run_sga(landscape; pop_size=100, generations=500, mutation_rate=0.01, tournament_size=3)

    println("Dataset: $(landscape.name)")
    println("Features: $(landscape.n_features)")
    println("Best fitness: $(best_fitness)")
    println("Best individual: $(best_individual)")
    println("Final generation max: $(last(max_history))")
    println("Final generation mean: $(last(mean_history))")
    println("Final generation min: $(last(min_history))")
    println("Final generation entropy: $(last(entropy_history))")

    pareto_individuals, pareto_objectives, front_size_history, nsga_entropy_history =
        run_nsga2(landscape; pop_size=100, generations=500, mutation_rate=0.01)

    println()
    println("NSGA-II")
    print_pareto_summary(pareto_individuals, pareto_objectives)
    println("Final generation front size: $(last(front_size_history))")
    println("Final generation entropy: $(last(nsga_entropy_history))")
end

main(ARGS)
