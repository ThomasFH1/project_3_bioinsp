using Pkg

project_dir = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(project_dir)

using Plots
using Project3Bioinsp

function triangle_landscape(; n::Int=16, m::Int=1, s::Int=4)
    values = [
        triangle_fitness(BitVector(((index >> (i - 1)) & 1) == 1 for i in 1:n); n=n, m=m, s=s)
        for index in 1:(2^n - 1)
    ]

    return Landscape(Float64.(values), n, "triangle", 0.0)
end

input = isempty(ARGS) ? joinpath(project_dir, "data", "01-breast-w_lr_F.h5") : ARGS[1]

landscape = if lowercase(input) == "triangle"
    triangle_landscape()
else
    dataset_path = isabspath(input) ? input : normpath(joinpath(project_dir, input))
    load_landscape(dataset_path; ε=0.1, name=splitext(basename(dataset_path))[1])
end

generations = 120
snapshot_intervals = 4
trace_generations = unique(round.(Int, range(0, generations; length=snapshot_intervals + 1)))

_, _, _, _, snapshots = run_nsga2(
    landscape;
    pop_size=100,
    generations=generations,
    mutation_rate=0.01,
    trace_generations=trace_generations,
)

plt = plot(
    xlabel="Selected features",
    ylabel=landscape.name == "triangle" ? "Fitness" : "Accuracy",
    title="NSGA-II frontier movement: $(landscape.name)",
    xlims=(0, landscape.n_features + 1),
    legend=:bottomright,
)

jitter_offsets = collect(range(-0.12, 0.12; length=length(snapshots)))

for (i, snapshot) in enumerate(snapshots)
    front_objectives = snapshot.objectives[snapshot.front]
    x = [-objective[2] + jitter_offsets[i] for objective in front_objectives]
    y = [objective[1] for objective in front_objectives]

    scatter!(
        plt,
        x,
        y;
        label="gen $(snapshot.generation)",
        markersize=4,
        alpha=0.8,
    )
end

plots_dir = joinpath(project_dir, "plots")
mkpath(plots_dir)
output_path = joinpath(plots_dir, "nsga2_frontier_$(landscape.name).png")
savefig(plt, output_path)
println("Saved $(output_path)")
