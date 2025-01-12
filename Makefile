run:
	go run .

preview: run
	cat output/graph.dot | dot -Tpng > output/graph.png