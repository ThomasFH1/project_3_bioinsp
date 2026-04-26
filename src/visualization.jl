using Plots

function visualize_landscape(landscape::Landscape; highlight_optima::Bool=true)
    indices = collect(1:length(landscape.values))
    x = count_ones.(indices)
    y = [fitness(index_to_bitstring(index, landscape.n_features), landscape) for index in indices]

    plt = scatter(
        x,
        y;
        xlabel="Active features",
        ylabel="Fitness",
        title=landscape.name,
        label="Solutions",
        alpha=0.5,
        markersize=3,
    )

    if highlight_optima
        optima = local_optima_mask(landscape)
        scatter!(
            plt,
            x[optima],
            y[optima];
            color=:red,
            label="Local optima",
            markersize=4,
        )
    end

    return plt
end

split_bits(n_bits::Int) = (div(n_bits, 2), n_bits - div(n_bits, 2))
gray_code(i::Int) = xor(i, i >> 1)

function gray_position_map(n_bits::Int)
    positions = zeros(Int, 1 << n_bits)
    for position in 0:((1 << n_bits) - 1)
        positions[gray_code(position) + 1] = position
    end
    return positions
end

function coordinates(index::Int, n_bits::Int; layout::Symbol=:normal)
    x_bits, y_bits = split_bits(n_bits)
    x_mask = (1 << x_bits) - 1
    x = index & x_mask
    y = index >> x_bits

    if layout == :gray
        x = gray_position_map(x_bits)[x + 1]
        y = gray_position_map(y_bits)[y + 1]
    elseif layout != :normal
        error("Unknown layout: $layout")
    end

    return x, y
end

function bitstring_label(bits::AbstractVector{Bool})
    # Print high-order bits first, even though algorithms store bits low-order first.
    return join(reverse([bit ? "1" : "0" for bit in bits]))
end

function minmax_scale(values::AbstractVector{<:Real})
    finite_values = [Float64(v) for v in values if isfinite(v)]
    isempty(finite_values) && return fill(NaN, length(values))

    lo = minimum(finite_values)
    hi = maximum(finite_values)
    if hi == lo
        return [isfinite(v) ? 0.0 : NaN for v in values]
    end

    return [isfinite(v) ? (Float64(v) - lo) / (hi - lo) : NaN for v in values]
end

function feature_landscape_scores(landscape::Landscape; contrast_power::Int=4)
    raw_accuracy = landscape.values
    selected = [count_ones(index) for index in eachindex(raw_accuracy)]
    penalty = landscape.penalty_weight .* selected
    fitness_values = raw_accuracy .- penalty
    normalized_accuracy = minmax_scale(raw_accuracy)
    visual_score = minmax_scale(fitness_values)

    return (
        raw_accuracy=raw_accuracy,
        normalized_accuracy=normalized_accuracy,
        selected=selected,
        penalty=penalty,
        fitness=fitness_values,
        visual_score=visual_score,
        color_value=visual_score .^ contrast_power,
        raw_optimum=argmax(raw_accuracy),
        fitness_optimum=argmax(fitness_values),
    )
end

function weight_landscape_scores(landscape::WeightLookupLandscape; contrast_power::Int=4)
    n_values = 1 << landscape.n_features
    fitness_values = [
        Float64(landscape.fitness_by_weight[count_ones(index) + 1])
        for index in 0:(n_values - 1)
    ]
    visual_score = minmax_scale(fitness_values)

    return (
        fitness=fitness_values,
        visual_score=visual_score,
        color_value=visual_score .^ contrast_power,
        global_optima=findall(==(maximum(fitness_values)), fitness_values) .- 1,
    )
end

function build_grid(values::AbstractVector{<:Real}, n_bits::Int;
                    include_zero::Bool=false, layout::Symbol=:normal)
    x_bits, y_bits = split_bits(n_bits)
    grid = fill(NaN, 1 << y_bits, 1 << x_bits)

    x_gray = layout == :gray ? gray_position_map(x_bits) : nothing
    y_gray = layout == :gray ? gray_position_map(y_bits) : nothing
    first_index = include_zero ? 0 : 1
    last_index = include_zero ? length(values) - 1 : length(values)
    x_mask = (1 << x_bits) - 1

    for index in first_index:last_index
        value_index = include_zero ? index + 1 : index
        x = index & x_mask
        y = index >> x_bits
        if layout == :gray
            x = x_gray[x + 1]
            y = y_gray[y + 1]
        elseif layout != :normal
            error("Unknown layout: $layout")
        end
        grid[y + 1, x + 1] = Float64(values[value_index])
    end

    return grid
end

function local_optima_indices(values::AbstractVector{<:Real}, n_bits::Int;
                              include_zero::Bool=false, strict::Bool=true)
    optima = Int[]
    first_index = include_zero ? 0 : 1
    last_index = include_zero ? length(values) - 1 : length(values)

    for index in first_index:last_index
        value_index = include_zero ? index + 1 : index
        current = Float64(values[value_index])
        isfinite(current) || continue

        is_optimum = true
        for bit in 0:(n_bits - 1)
            neighbor = xor(index, 1 << bit)
            if neighbor < first_index || neighbor > last_index
                continue
            end

            neighbor_index = include_zero ? neighbor + 1 : neighbor
            neighbor_value = Float64(values[neighbor_index])
            if strict ? neighbor_value >= current : neighbor_value > current
                is_optimum = false
                break
            end
        end

        is_optimum && push!(optima, index)
    end

    return optima
end

function local_optima_weights(landscape::WeightLookupLandscape; strict::Bool=true)
    optima = Int[]
    values = Float64.(landscape.fitness_by_weight)

    for k in 0:landscape.n_features
        current = values[k + 1]
        lower_ok = k == 0 || (strict ? values[k] < current : values[k] <= current)
        upper_ok = k == landscape.n_features || (strict ? values[k + 2] < current : values[k + 2] <= current)
        lower_ok && upper_ok && push!(optima, k)
    end

    return optima
end

function base_contour_plot(grid; title::String, xlabel::String, ylabel::String,
                           levels::Int, contrast_power::Int,
                           filled::Bool=true, heatmap_style::Bool=false)
    plot_function = heatmap_style ? heatmap : (filled ? contourf : contour)
    contour_levels = filled ?
        collect(range(0.0, 1.0; length=levels)) :
        collect(range(0.0, 1.0; length=levels + 2))[2:end-1]

    return plot_function(
        0:(size(grid, 2) - 1),
        0:(size(grid, 1) - 1),
        grid;
        levels=heatmap_style ? levels : contour_levels,
        color=:viridis,
        clim=(0, 1),
        xlims=(-0.5, size(grid, 2) - 0.5),
        ylims=(-0.5, size(grid, 1) - 0.5),
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        colorbar_title="min-max scaled fitness ^ $contrast_power",
        aspect_ratio=:equal,
        size=(1800, 1400),
        legend=:outertopright,
    )
end

function axis_labels(n_bits::Int, layout::Symbol)
    x_bits, y_bits = split_bits(n_bits)
    order = layout == :gray ? "Gray-code order" : "binary order"
    return "x = low-order bits 1:$x_bits ($order)",
           "y = high-order bits $(x_bits + 1):$(x_bits + y_bits) ($order)"
end

function coord_vectors(indices, n_bits::Int; layout::Symbol)
    points = [coordinates(index, n_bits; layout=layout) for index in indices]
    return first.(points), last.(points)
end

function overlay_path!(plt, path_bits; layout::Symbol=:normal, label_every::Int=10)
    isempty(path_bits) && return plt

    n_bits = length(first(path_bits))
    indices = [bitstring_to_index(bits) for bits in path_bits]
    xs, ys = coord_vectors(indices, n_bits; layout=layout)

    plot!(plt, xs, ys; color=:black, linewidth=1.2, label="algorithm path")
    scatter!(plt, xs, ys; color=:black, markersize=2.5, label=false)
    scatter!(plt, [first(xs)], [first(ys)]; color=:red, markersize=7, marker=:circle, label="start")
    scatter!(plt, [last(xs)], [last(ys)]; color=:magenta, markersize=9, marker=:star5, label="finish")

    label_indices = Int[1]
    previous = (first(xs), first(ys))
    for i in 2:length(xs)
        current = (xs[i], ys[i])
        if current != previous && i - last(label_indices) >= max(1, label_every)
            push!(label_indices, i)
        end
        previous = current
    end
    last(label_indices) == length(xs) || push!(label_indices, length(xs))

    scatter!(
        plt,
        xs[label_indices] .+ 0.35,
        ys[label_indices] .+ 0.35;
        color=:white,
        markerstrokecolor=:white,
        markersize=5,
        markeralpha=0.75,
        label=false,
    )
    for i in label_indices
        annotate!(plt, xs[i] + 0.35, ys[i] + 0.35, text(string(i), 6, :black))
    end

    return plt
end

function plot_feature_landscape_map(landscape::Landscape; layout::Symbol=:normal,
                                    levels::Int=15, scores=nothing,
                                    local_optima=nothing, path_bits=nothing,
                                    label_every::Int=10, contrast_power::Int=4,
                                    filled::Bool=true, heatmap_style::Bool=false)
    scores === nothing && (scores = feature_landscape_scores(landscape))
    local_optima === nothing && (local_optima = local_optima_indices(scores.fitness, landscape.n_features; strict=true))

    grid = build_grid(scores.color_value, landscape.n_features; include_zero=false, layout=layout)
    xlabel, ylabel = axis_labels(landscape.n_features, layout)
    map_kind = landscape.penalty_weight == 0 ? "accuracy" : "penalized fitness"
    plt = base_contour_plot(
        grid;
        title="$(landscape.name): $(map_kind), $(layout) layout",
        xlabel=xlabel,
        ylabel=ylabel,
        levels=levels,
        contrast_power=contrast_power,
        filled=filled,
        heatmap_style=heatmap_style,
    )

    if !isempty(local_optima)
        local_marker_size = length(local_optima) > 2000 ? 1.0 : 4
        local_marker_alpha = length(local_optima) > 2000 ? 0.55 : 0.9
        xs, ys = coord_vectors(local_optima, landscape.n_features; layout=layout)
        scatter!(plt, xs, ys; color=:magenta, marker=:cross,
                 markersize=local_marker_size, markeralpha=local_marker_alpha,
                 label="strict local optima")
    end

    fitness_bits = index_to_bitstring(scores.fitness_optimum, landscape.n_features)
    raw_bits = index_to_bitstring(scores.raw_optimum, landscape.n_features)
    fx, fy = coordinates(scores.fitness_optimum, landscape.n_features; layout=layout)
    rx, ry = coordinates(scores.raw_optimum, landscape.n_features; layout=layout)

    if scores.fitness_optimum == scores.raw_optimum
        scatter!(plt, [fx], [fy]; color=:red, marker=:star5, markersize=11,
                 label="global optimum ($(bitstring_label(fitness_bits)))")
    else
        scatter!(plt, [fx], [fy]; color=:red, marker=:star5, markersize=11,
                 label="fitness optimum ($(bitstring_label(fitness_bits)))")
        scatter!(plt, [rx], [ry]; color=:cyan, marker=:xcross, markersize=10,
                 label="raw accuracy optimum ($(bitstring_label(raw_bits)))")
    end

    path_bits !== nothing && overlay_path!(plt, path_bits; layout=layout, label_every=label_every)
    return plt
end

function plot_weight_landscape_map(landscape::WeightLookupLandscape; layout::Symbol=:normal,
                                   levels::Int=15, scores=nothing,
                                   local_optima=nothing, path_bits=nothing,
                                   label_every::Int=10, contrast_power::Int=4,
                                   filled::Bool=true, heatmap_style::Bool=false)
    if landscape.n_features > 20
        error("Full contour maps are disabled for n > 20. Use Hamming-weight plots instead.")
    end

    scores === nothing && (scores = weight_landscape_scores(landscape))
    local_optima === nothing && (local_optima = local_optima_indices(scores.fitness, landscape.n_features; include_zero=true, strict=true))

    grid = build_grid(scores.color_value, landscape.n_features; include_zero=true, layout=layout)
    xlabel, ylabel = axis_labels(landscape.n_features, layout)
    plt = base_contour_plot(
        grid;
        title="$(landscape.name): synthetic fitness, $(layout) layout",
        xlabel=xlabel,
        ylabel=ylabel,
        levels=levels,
        contrast_power=contrast_power,
        filled=filled,
        heatmap_style=heatmap_style,
    )

    if !isempty(local_optima)
        local_marker_size = length(local_optima) > 2000 ? 1.0 : 4
        local_marker_alpha = length(local_optima) > 2000 ? 0.55 : 0.9
        xs, ys = coord_vectors(local_optima, landscape.n_features; layout=layout)
        scatter!(plt, xs, ys; color=:magenta, marker=:cross,
                 markersize=local_marker_size, markeralpha=local_marker_alpha,
                 label="strict local optima")
    end

    xs, ys = coord_vectors(scores.global_optima, landscape.n_features; layout=layout)
    scatter!(plt, xs, ys; color=:red, marker=:star5, markersize=5,
             markeralpha=0.9, label="global optima")

    path_bits !== nothing && overlay_path!(plt, path_bits; layout=layout, label_every=label_every)
    return plt
end

function plot_weight_lookup(landscape::WeightLookupLandscape)
    weights = 0:landscape.n_features
    values = Float64.(landscape.fitness_by_weight)
    local_weights = local_optima_weights(landscape; strict=true)
    global_weights = findall(==(maximum(values)), values) .- 1

    plt = plot(
        weights,
        values;
        marker=:circle,
        linewidth=2,
        xlabel="Active bits",
        ylabel="Fitness",
        title="$(landscape.name): fitness by Hamming weight",
        label="fitness",
        size=(1500, 900),
    )

    scatter!(plt, local_weights, values[local_weights .+ 1];
             color=:orange, marker=:diamond, markersize=6, label="strict local optima")
    scatter!(plt, global_weights, values[global_weights .+ 1];
             color=:dodgerblue, marker=:star5, markersize=8, label="global optima")

    return plt
end

function plot_weight_path(path_bits, landscape::WeightLookupLandscape; algorithm::String)
    steps = 1:length(path_bits)
    weights = [count(bits) for bits in path_bits]
    fitnesses = [fitness(bits, landscape) for bits in path_bits]

    plt = plot(
        steps,
        fitnesses;
        color=:black,
        linewidth=2,
        marker=:circle,
        markersize=3,
        xlabel="Generation / iteration",
        ylabel="Fitness",
        title="$(landscape.name): $algorithm path fitness",
        label="fitness",
        size=(1500, 900),
    )
    plot!(plt, steps, weights; color=:steelblue, linewidth=2, linestyle=:dash,
          ylabel="Fitness / active bits", label="active bits")

    return plt
end

function path_rows(path_bits, steps, landscape::Landscape; algorithm::String,
                   path_kind::String, scores=nothing)
    scores === nothing && (scores = feature_landscape_scores(landscape))
    rows = NamedTuple[]

    for (step, bits) in zip(steps, path_bits)
        index = bitstring_to_index(bits)
        x_normal, y_normal = coordinates(index, landscape.n_features; layout=:normal)
        x_gray, y_gray = coordinates(index, landscape.n_features; layout=:gray)

        if index == 0
            row = (
                step=step, algorithm=algorithm, path_kind=path_kind,
                bitstring=bitstring_label(bits), index=index,
                x_normal=x_normal, y_normal=y_normal, x_gray=x_gray, y_gray=y_gray,
                mean_accuracy=NaN, normalized_accuracy=NaN, selected_features=count(bits),
                penalty=NaN, fitness=-Inf, visual_score=NaN,
            )
        else
            row = (
                step=step, algorithm=algorithm, path_kind=path_kind,
                bitstring=bitstring_label(bits), index=index,
                x_normal=x_normal, y_normal=y_normal, x_gray=x_gray, y_gray=y_gray,
                mean_accuracy=scores.raw_accuracy[index],
                normalized_accuracy=scores.normalized_accuracy[index],
                selected_features=scores.selected[index],
                penalty=scores.penalty[index],
                fitness=scores.fitness[index],
                visual_score=scores.visual_score[index],
            )
        end
        push!(rows, row)
    end

    return rows
end

function path_rows(path_bits, steps, landscape::WeightLookupLandscape; algorithm::String,
                   path_kind::String)
    values = Float64.(landscape.fitness_by_weight)
    rows = NamedTuple[]
    min_value = minimum(values)
    max_value = maximum(values)

    for (step, bits) in zip(steps, path_bits)
        index = bitstring_to_index(bits)
        fit = fitness(bits, landscape)
        visual_score = max_value == min_value ? 0.0 : (fit - min_value) / (max_value - min_value)
        x_normal, y_normal = coordinates(index, landscape.n_features; layout=:normal)
        x_gray, y_gray = coordinates(index, landscape.n_features; layout=:gray)
        push!(
            rows,
            (
                step=step, algorithm=algorithm, path_kind=path_kind,
                bitstring=bitstring_label(bits), index=index,
                x_normal=x_normal, y_normal=y_normal, x_gray=x_gray, y_gray=y_gray,
                triangle_fitness=fit, hamming_weight=count(bits), visual_score=visual_score,
            ),
        )
    end

    return rows
end

function csv_value(value)
    if value isa AbstractString
        escaped = replace(value, "\"" => "\"\"")
        return occursin(",", escaped) ? "\"$escaped\"" : escaped
    end
    return string(value)
end

function save_path_table(path::String, rows)
    isempty(rows) && return
    headers = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(headers, ","))
        for row in rows
            println(io, join([csv_value(getproperty(row, header)) for header in headers], ","))
        end
    end
end
