module Project3Bioinsp

export Landscape, load_landscape, apply_penalty, local_optima_mask, accuracy
export WeightLookupLandscape, triangle_fitness, triangle_landscape, test_triangle_landscape
export run_sga, run_nsga2, run_pso
export visualize_landscape, feature_landscape_scores, weight_landscape_scores
export local_optima_indices, local_optima_weights
export plot_feature_landscape_map, plot_weight_landscape_map, plot_weight_lookup, plot_weight_path
export path_rows, save_path_table
export index_to_bitstring

include("landscapes.jl")
include("triangle.jl")
include("sga.jl")
include("nsga2/helpers.jl")
include("nsga2/selectors.jl")
include("nsga2/nsga2.jl")
include("pso.jl")
include("visualization.jl")

end
