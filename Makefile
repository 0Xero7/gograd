run:
	go run .

preview: run
	export OPENBLAS_NUM_THREADS=8  # Use 4 threads (adjust to your CPU core count)
	export OMP_NUM_THREADS=8       # If using OpenMP
	cat output/graph.dot | dot -Tpng > output/graph.png