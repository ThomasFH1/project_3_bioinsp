JULIA ?= julia
PROJECT ?= --project=.
SEED ?= 1
GENERATIONS ?= 500
POP_SIZE ?= 100
MUTATION_RATE ?= 0.01
EPSILON ?= 0.1
TEST_EPSILON ?= 0.0
TRACE ?= false
TRACE_INTERVAL ?= 50
TRACE_VERBOSE ?= false
N_SEEDS ?= 10
STATS_DATASETS ?= data/01-breast-w_lr_F.h5 data/05-credit-a_rf_F.h5 data/08-letter-r_knn_F.h5

RUN_NSGA2 = $(JULIA) $(PROJECT) scripts/run_nsga2_dataset.jl
RUN_TRIANGLE = $(JULIA) $(PROJECT) scripts/run_test_triangle_nsga2.jl
RUN_NSGA2_STATS = $(JULIA) $(PROJECT) scripts/run_nsga2_stats.jl

.PHONY: breast credit zoo letter hepatitis all-data triangle triangle-verbose nsga-stats nsga-train-triangle-stats fourier

breast:
	$(RUN_NSGA2) data/01-breast-w_lr_F.h5 $(SEED) $(GENERATIONS) $(POP_SIZE) $(MUTATION_RATE) $(EPSILON) $(TRACE) $(TRACE_INTERVAL) $(TRACE_VERBOSE)

credit:
	$(RUN_NSGA2) data/05-credit-a_rf_F.h5 $(SEED) $(GENERATIONS) $(POP_SIZE) $(MUTATION_RATE) $(EPSILON) $(TRACE) $(TRACE_INTERVAL) $(TRACE_VERBOSE)

zoo:
	$(RUN_NSGA2) data/06-zoo_lr_F.h5 $(SEED) $(GENERATIONS) $(POP_SIZE) $(MUTATION_RATE) $(TEST_EPSILON) $(TRACE) $(TRACE_INTERVAL) $(TRACE_VERBOSE)

letter:
	$(RUN_NSGA2) data/08-letter-r_knn_F.h5 $(SEED) $(GENERATIONS) $(POP_SIZE) $(MUTATION_RATE) $(EPSILON) $(TRACE) $(TRACE_INTERVAL) $(TRACE_VERBOSE)

hepatitis:
	$(RUN_NSGA2) data/10-hepatitis_lr_F.h5 $(SEED) $(GENERATIONS) $(POP_SIZE) $(MUTATION_RATE) $(TEST_EPSILON) $(TRACE) $(TRACE_INTERVAL) $(TRACE_VERBOSE)

all-data: breast credit zoo letter hepatitis

triangle:
	$(RUN_TRIANGLE) $(SEED) $(GENERATIONS) $(TRACE_INTERVAL) false

triangle-verbose:
	$(RUN_TRIANGLE) $(SEED) $(GENERATIONS) 1 true

nsga-stats:
	$(RUN_NSGA2_STATS) $(N_SEEDS) $(GENERATIONS) $(POP_SIZE) $(MUTATION_RATE) $(EPSILON) $(STATS_DATASETS)

nsga-train-triangle-stats:
	$(RUN_NSGA2_STATS) $(N_SEEDS) $(GENERATIONS) $(POP_SIZE) $(MUTATION_RATE) 0.0 train_triangle

fourier:
	$(JULIA) $(PROJECT) scripts/fourier_analysis.jl
