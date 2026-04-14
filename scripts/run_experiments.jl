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
end

main(ARGS)
