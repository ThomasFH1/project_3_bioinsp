using Pkg

project_dir = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(project_dir)

using Plots
using Project3Bioinsp

dataset_path = isempty(ARGS) ? joinpath(project_dir, "data", "06-zoo_lr_F.h5") : ARGS[1]
dataset_path = isabspath(dataset_path) ? dataset_path : normpath(joinpath(project_dir, dataset_path))

landscape = load_landscape(dataset_path; ε=0.1, name=splitext(basename(dataset_path))[1])
indices = collect(eachindex(landscape.values))
x = count_ones.(indices)
y = [
    apply_penalty(landscape.values[index], count_ones(index), landscape.penalty_weight)
    for index in indices
]

optima = local_optima_mask(landscape)
pareto_individuals, _, _, _ = run_nsga2(
    landscape;
    pop_size=100,
    generations=120,
    mutation_rate=0.01,
)
pareto_x = count.(pareto_individuals)
pareto_y = [
    Project3Bioinsp.fitness(individual, landscape)
    for individual in pareto_individuals
]

plt = scatter(
    x,
    y;
    xlabel="Selected features",
    ylabel="Penalized fitness",
    title="Landscape + local optima + NSGA-II: $(landscape.name)",
    color=:gray,
    alpha=0.18,
    markersize=2,
    label="Landscape",
    xlims=(0, landscape.n_features + 1),
)

scatter!(
    plt,
    x[optima],
    y[optima];
    color=:red,
    markersize=4,
    label="Local optima",
)

scatter!(
    plt,
    pareto_x,
    pareto_y;
    color=:blue,
    marker=:star5,
    markersize=8,
    label="NSGA-II final F1",
)

plots_dir = joinpath(project_dir, "plots")
mkpath(plots_dir)
output_path = joinpath(plots_dir, "nsga2_overlay_$(landscape.name).png")
savefig(plt, output_path)
println("Saved $(output_path)")
