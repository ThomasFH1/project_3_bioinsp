using Pkg

project_dir = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(project_dir)

using Plots
using Statistics
using Project3Bioinsp

plots_dir = joinpath(project_dir, "plots")
mkpath(plots_dir)

# ── Load all landscapes ────────────────────────────────────────────────────────

data_dir = joinpath(project_dir, "data")
h5_files = sort(filter(p -> endswith(p, ".h5"), readdir(data_dir; join=true)))

landscapes = [load_landscape(p; ε=0.1, name=splitext(basename(p))[1]) for p in h5_files]


println("Loaded $(length(landscapes)) landscapes: $(join([l.name for l in landscapes], ", "))")

# ── Run PSO on all landscapes (10 runs each for statistics) ───────────────────

N_RUNS = 10

# Store results per landscape
all_max_histories   = Dict{String, Vector{Vector{Float64}}}()
all_mean_histories  = Dict{String, Vector{Vector{Float64}}}()
all_min_histories   = Dict{String, Vector{Vector{Float64}}}()
all_ent_histories   = Dict{String, Vector{Vector{Float64}}}()
all_best_fits       = Dict{String, Vector{Float64}}()

for landscape in landscapes
    println("\nRunning PSO on $(landscape.name) ($N_RUNS runs)...")

    max_hs  = Vector{Float64}[]
    mean_hs = Vector{Float64}[]
    min_hs  = Vector{Float64}[]
    ent_hs  = Vector{Float64}[]
    best_fs = Float64[]

    for run in 1:N_RUNS
        best_pos, best_fit, max_h, mean_h, min_h, ent_h =
            run_pso(landscape; swarm_size=50, iterations=500)
        push!(max_hs,  max_h)
        push!(mean_hs, mean_h)
        push!(min_hs,  min_h)
        push!(ent_hs,  ent_h)
        push!(best_fs, best_fit)
        print("  run $run: best=$(round(best_fit, digits=4))\n")
    end

    all_max_histories[landscape.name]  = max_hs
    all_mean_histories[landscape.name] = mean_hs
    all_min_histories[landscape.name]  = min_hs
    all_ent_histories[landscape.name]  = ent_hs
    all_best_fits[landscape.name]      = best_fs
end

# ── Plot 1: Fitness over time (max/mean/min) for each landscape ───────────────
# Shows how PSO converges — one plot per landscape, averaged over all runs

println("\nGenerating Plot 1: Fitness over time...")

for landscape in landscapes
    name      = landscape.name
    max_hs    = all_max_histories[name]
    mean_hs   = all_mean_histories[name]
    min_hs    = all_min_histories[name]
    iters     = 1:length(max_hs[1])

    avg_max  = [mean([max_hs[r][i]  for r in 1:N_RUNS]) for i in iters]
    avg_mean = [mean([mean_hs[r][i] for r in 1:N_RUNS]) for i in iters]
    avg_min  = [mean([min_hs[r][i]  for r in 1:N_RUNS]) for i in iters]

    plt = plot(iters, avg_max;
        label="Max fitness",
        color=:blue,
        linewidth=2,
        xlabel="Iteration",
        ylabel="Fitness",
        title="PSO Fitness over Time\n$(name)",
        legend=:bottomright)

    plot!(plt, iters, avg_mean;
        label="Mean fitness",
        color=:green,
        linewidth=2,
        linestyle=:dash)

    plot!(plt, iters, avg_min;
        label="Min fitness",
        color=:red,
        linewidth=2,
        linestyle=:dot)

    savefig(plt, joinpath(plots_dir, "pso_fitness_$(name).png"))
    println("  Saved pso_fitness_$(name).png")
end

# ── Plot 2: Entropy over time for each landscape ──────────────────────────────
# Shows how swarm diversity changes — high entropy = exploring, low = converged

println("\nGenerating Plot 2: Entropy over time...")

plt_ent = plot(;
    xlabel="Iteration",
    ylabel="Entropy",
    title="PSO Swarm Entropy over Time\n(averaged over $N_RUNS runs)",
    legend=:topright)

colors = [:blue, :red, :green, :orange]

for (idx, landscape) in enumerate(landscapes)
    name   = landscape.name
    ent_hs = all_ent_histories[name]
    iters  = 1:length(ent_hs[1])

    avg_ent = [mean([ent_hs[r][i] for r in 1:N_RUNS]) for i in iters]

    plot!(plt_ent, iters, avg_ent;
        label=name,
        color=colors[idx],
        linewidth=2)
end

savefig(plt_ent, joinpath(plots_dir, "pso_entropy_all.png"))
println("  Saved pso_entropy_all.png")

# ── Plot 3: Best fitness comparison across landscapes (box-like bar chart) ────
# Shows mean ± std of best fitness found over 10 runs per landscape

println("\nGenerating Plot 3: Best fitness comparison...")

names     = [l.name for l in landscapes]
means     = [mean(all_best_fits[n]) for n in names]
stds      = [std(all_best_fits[n])  for n in names]

plt_comp = bar(names, means;
    yerror=stds,
    xlabel="Landscape",
    ylabel="Best Fitness (mean ± std)",
    title="PSO Best Fitness across Landscapes\n($N_RUNS runs each)",
    legend=false,
    color=:steelblue,
    xrotation=15)

savefig(plt_comp, joinpath(plots_dir, "pso_comparison.png"))
println("  Saved pso_comparison.png")

# ── Print summary statistics table ────────────────────────────────────────────

println("\n═══════════════════════════════════════════════════════")
println("PSO Results Summary ($N_RUNS runs per landscape)")
println("═══════════════════════════════════════════════════════")
println("$(rpad("Landscape", 30)) $(rpad("Mean", 10)) $(rpad("Std", 10)) $(rpad("Best", 10))")
println("─"^62)
for landscape in landscapes
    name = landscape.name
    fs   = all_best_fits[name]
    println("$(rpad(name, 30)) $(rpad(round(mean(fs), digits=4), 10)) $(rpad(round(std(fs), digits=4), 10)) $(round(maximum(fs), digits=4))")
end
println("═"^62)
