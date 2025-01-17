package main

import (
	"flag"
	mnisttensor "gograd/trainsets/mnist_tensor"
	"math/rand"
	_ "net/http/pprof"
)

var memprofile = flag.String("memprofile", "", "write memory profile to file")

/*
BEFORE PERF OPT: (50, 512, 0.00001)
	Alloc = 5320 MB
	Total Alloc = 351473 MB
	Epoch = 49, Loss = [0.288718148459]
	Forward Pass Time = 2.231000
	Backward Pass Time = 1.846000
	Update Time = 0.046000
	Epoch 49 completed in 4.123000. [3749.100000 per epoch].
	Train accuracy: [28913] of [32000] = 0.90
	Test accuracy: [8932] of [10000] = 0.89

"Optimize" Add function:
	Alloc = 5326 MB
	Total Alloc = 351795 MB
	Epoch = 49, Loss = [0.324278995499]
	Forward Pass Time = 1.798000
	Backward Pass Time = 1.657000
	Update Time = 0.047000
	Epoch 49 completed in 3.502000. [3523.980000 per epoch]

"Optimize" Mul function:
	Alloc = 5333 MB
	Total Alloc = 352118 MB
	Epoch = 49, Loss = [0.329019796407]
	Forward Pass Time = 1.814000
	Backward Pass Time = 1.417000
	Update Time = 0.050000
	Epoch 49 completed in 3.281000. [2934.000000 per epoch].

"Pool + SIMD":
	Alloc = 5349 MB
	Total Alloc = 333999 MB
	Epoch = 49, Loss = [0.329019796407]
	Forward Pass Time = 1.164000
	Backward Pass Time = 0.981000
	Update Time = 0.074000
	Epoch 49 completed in 2.219000. [2238.100000 per epoch].

"Optimized Pool":
	Alloc = 5349 MB
	Total Alloc = 118595 MB
	Epoch = 49, Loss = [0.329019796407]
	Forward Pass Time = 0.825000
	Backward Pass Time = 0.847000
	Update Time = 0.066000
	Epoch 49 completed in 1.738000. [1809.320000 per epoch].
*/

func main() {
	flag.Parse()

	rand.Seed(1337)

	// iristensor.LoadAndSplitDataset()
	// mlp2 := iristensor.TrainIris(5000, 100, 0.001)
	// iristensor.TestIris(mlp2)

	mnisttensor.LoadDataset()
	// f, _ := os.Create("cpu.prof")
	// defer f.Close() // error handling omitted for example

	// if err := pprof.StartCPUProfile(f); err != nil {
	// 	log.Fatal("could not start CPU profile: ", err)
	// }
	mlp3 := mnisttensor.TrainMNIST(50, 512, 0.00001)
	// pprof.StopCPUProfile()

	mnisttensor.TestMNIST(mlp3)
}
