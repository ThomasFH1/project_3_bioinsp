using Pkg

project_dir = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(project_dir)

using Plots
using Project3Bioinsp

function resolve_dataset_paths(project_dir::String, args::Vector{String})
    data_dir = joinpath(project_dir, "data")

    if isempty(args)
        return sort(filter(path -> endswith(path, ".h5"), readdir(data_dir; join=true)))
    end

    return [
        isabspath(path) ? path : normpath(joinpath(project_dir, path))
        for path in args
    ]
end

function main(args::Vector{String})
    dataset_paths = resolve_dataset_paths(project_dir, args)
    plots_dir = joinpath(project_dir, "plots")
    mkpath(plots_dir)

    for dataset_path in dataset_paths
        landscape_name = splitext(basename(dataset_path))[1]
        landscape = load_landscape(dataset_path; ε=0.1, name=landscape_name)
        plt = visualize_landscape(landscape; highlight_optima=true)

        output_path = joinpath(plots_dir, "$(landscape_name)_landscape.png")
        savefig(plt, output_path)
        println("Saved $(output_path)")
    end
end

main(ARGS)
