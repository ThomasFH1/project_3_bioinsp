module Project3Bioinsp

export load_landscape, apply_penalty
export triangle_fitness
export run_sga, run_nsga2, run_pso
export visualize_landscape

include("landscapes.jl")
include("triangle.jl")
include("sga.jl")
include("nsga2.jl")
include("pso.jl")
include("visualization.jl")

end
