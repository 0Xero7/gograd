run:
	export OPENBLAS_NUM_THREADS=4  # Use 4 threads (adjust to your CPU core count)
	export OMP_NUM_THREADS=4      # If using OpenMP
	CGO_LDFLAGS="-L/usr/lib/x86_64-linux-gnu/openblas-pthread -lopenblas" go run -tags=openblas .

preview: run
	export OPENBLAS_NUM_THREADS=4  # Use 4 threads (adjust to your CPU core count)
	export OMP_NUM_THREADS=4      # If using OpenMP
	cat output/graph.dot | dot -Tpng > output/graph.png