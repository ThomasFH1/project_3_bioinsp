using Pkg

project_dir = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(project_dir)

using Plots
using Statistics
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

function csv_escape(value)
    text = string(value)
    if occursin(",", text) || occursin("\"", text) || occursin("\n", text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_csv(path::String, header, rows)
    open(path, "w") do io
        println(io, join(csv_escape.(header), ","))
        for row in rows
            println(io, join(csv_escape.(row), ","))
        end
    end
end

function objective_specs()
    return [
        ("raw_accuracy", false),
        ("penalized_fitness", true),
    ]
end

function nonconstant_ratio(energy::Vector{Float64}, order::Int)
    nonconstant_energy = sum(energy[2:end])
    if nonconstant_energy == 0.0
        return 0.0
    end
    return energy[order + 1] / nonconstant_energy
end

function order_energy_rows(energy::Vector{Float64})
    total_energy = sum(energy)
    nonconstant_energy = sum(energy[2:end])
    cumulative_nonconstant = 0.0
    rows = []

    for order in 0:(length(energy) - 1)
        if order > 0
            cumulative_nonconstant += energy[order + 1]
        end

        push!(
            rows,
            (
                order,
                energy[order + 1],
                total_energy == 0.0 ? 0.0 : energy[order + 1] / total_energy,
                order == 0 || nonconstant_energy == 0.0 ? 0.0 : energy[order + 1] / nonconstant_energy,
                nonconstant_energy == 0.0 ? 0.0 : cumulative_nonconstant / nonconstant_energy,
            ),
        )
    end

    return rows
end

function main_effect_rows(coefficients)
    effects = feature_main_effects(coefficients)
    rows = [
        (
            feature,
            coefficients[2^(feature - 1) + 1],
            effects[feature],
            abs(effects[feature]),
        )
        for feature in eachindex(effects)
    ]
    sort!(rows; by=row -> row[4], rev=true)
    return rows
end

function pair_rows(coefficients)
    rows = [
        (
            left,
            right,
            coefficient,
            interaction_effect,
            abs(interaction_effect),
        )
        for (left, right, coefficient, interaction_effect) in pair_interaction_effects(coefficients)
    ]
    sort!(rows; by=row -> row[5], rev=true)
    return rows
end

function top_term_rows(coefficients, n_features::Int; limit::Int=50)
    rows = []

    for mask in 1:(length(coefficients) - 1)
        features = mask_features(mask, n_features)
        push!(
            rows,
            (
                count_ones(mask),
                mask,
                join(features, "+"),
                coefficients[mask + 1],
                abs(coefficients[mask + 1]),
            ),
        )
    end

    sort!(rows; by=row -> row[5], rev=true)
    return rows[1:min(limit, length(rows))]
end

function plot_order_energy(output_path::String, landscape_name::String, objective_name::String, energy::Vector{Float64})
    n_features = length(energy) - 1
    ratios = [nonconstant_ratio(energy, order) for order in 1:n_features]

    plt = bar(
        1:n_features,
        ratios;
        xlabel="Interaction order",
        ylabel="Share of nonconstant Fourier energy",
        title="$(landscape_name) - $(objective_name)",
        label=false,
        color=:steelblue,
        xticks=1:n_features,
        size=(max(900, 45 * n_features), 550),
    )
    savefig(plt, output_path)
end

function plot_main_effects(output_path::String, landscape_name::String, objective_name::String, coefficients)
    effects = feature_main_effects(coefficients)
    order = collect(eachindex(effects))
    labels = ["F$(feature)" for feature in order]
    positions = collect(eachindex(labels))

    plt = bar(
        positions,
        effects[order];
        xlabel="Feature",
        ylabel="Mean selected - mean absent",
        title="$(landscape_name) - $(objective_name)",
        label=false,
        color=:navy,
        linecolor=:match,
        linewidth=0,
        bar_width=0.4,
        xticks=(positions, labels),
        xrotation=60,
        tickfontsize=8,
        size=(max(900, 45 * length(labels)), 550),
    )
    hline!(plt, [0.0]; color=:black, linewidth=1, label=false)
    savefig(plt, output_path)
end

function analyze_landscape(landscape::Landscape, output_dir::String)
    summary_rows = []

    for (objective_name, penalized) in objective_specs()
        values = full_landscape_values(landscape; penalized=penalized)
        coefficients = fourier_coefficients(landscape; penalized=penalized)
        energy = spectral_energy_by_order(coefficients)
        n_features = landscape.n_features
        nonconstant_energy = sum(energy[2:end])
        high_order_ratio = if n_features >= 3 && nonconstant_energy > 0.0
            sum(energy[4:end]) / nonconstant_energy
        else
            0.0
        end

        prefix = joinpath(output_dir, "$(landscape.name)_$(objective_name)")

        write_csv(
            "$(prefix)_order_energy.csv",
            [
                "order",
                "energy",
                "share_of_total_energy",
                "share_of_nonconstant_energy",
                "cumulative_share_of_nonconstant_energy",
            ],
            order_energy_rows(energy),
        )

        write_csv(
            "$(prefix)_main_effects.csv",
            ["feature", "coefficient", "main_effect", "abs_main_effect"],
            main_effect_rows(coefficients),
        )

        write_csv(
            "$(prefix)_pair_interactions.csv",
            ["feature_a", "feature_b", "coefficient", "interaction_effect", "abs_interaction_effect"],
            pair_rows(coefficients),
        )

        write_csv(
            "$(prefix)_top_terms.csv",
            ["order", "mask", "features", "coefficient", "abs_coefficient"],
            top_term_rows(coefficients, n_features),
        )

        plot_order_energy("$(prefix)_order_energy.png", landscape.name, objective_name, energy)
        plot_main_effects("$(prefix)_main_effects.png", landscape.name, objective_name, coefficients)

        push!(
            summary_rows,
            (
                landscape.name,
                objective_name,
                n_features,
                values[1],
                mean(values),
                std(values),
                minimum(values),
                maximum(values),
                coefficients[1],
                energy[1],
                nonconstant_energy,
                n_features >= 1 ? nonconstant_ratio(energy, 1) : 0.0,
                n_features >= 2 ? nonconstant_ratio(energy, 2) : 0.0,
                high_order_ratio,
            ),
        )

        strongest = first(main_effect_rows(coefficients), min(5, n_features))
        println("$(landscape.name) / $(objective_name)")
        for row in strongest
            direction = row[3] >= 0 ? "helps" : "hurts"
            println("  F$(row[1]) $(direction): main_effect=$(round(row[3], sigdigits=5))")
        end
    end

    return summary_rows
end

function main(args::Vector{String})
    dataset_paths = resolve_dataset_paths(project_dir, args)
    output_dir = joinpath(project_dir, "plots", "fourier")
    mkpath(output_dir)

    all_summary_rows = []

    for dataset_path in dataset_paths
        landscape_name = splitext(basename(dataset_path))[1]
        landscape = load_landscape(dataset_path; ε=0.1, name=landscape_name)
        append!(all_summary_rows, analyze_landscape(landscape, output_dir))
    end

    write_csv(
        joinpath(output_dir, "fourier_summary.csv"),
        [
            "landscape",
            "objective",
            "n_features",
            "imputed_empty_value",
            "mean_value",
            "std_value",
            "min_value",
            "max_value",
            "constant_coefficient",
            "constant_energy",
            "nonconstant_energy",
            "first_order_energy_ratio",
            "second_order_energy_ratio",
            "third_and_higher_order_energy_ratio",
        ],
        all_summary_rows,
    )

    println()
    println("Saved Fourier analysis outputs to $(output_dir)")
end

main(ARGS)
