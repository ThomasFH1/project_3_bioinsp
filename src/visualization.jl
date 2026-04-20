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
