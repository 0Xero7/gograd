package main

import (
	"flag"
	mnisttensor "gograd/trainsets/mnist_tensor"
	"log"
	"math/rand"
	_ "net/http/pprof"
	"os"
	"runtime/pprof"
)

var memprofile = flag.String("memprofile", "", "write memory profile to file")

func main() {
	flag.Parse()

	rand.Seed(1337)

	f, _ := os.Create("cpu.prof")
	defer f.Close() // error handling omitted for example
	if err := pprof.StartCPUProfile(f); err != nil {
		log.Fatal("could not start CPU profile: ", err)
	}

	// iristensor.LoadAndSplitDataset()
	// mlp2 := iristensor.TrainIris(5000, 100, 0.01)
	// iristensor.TestIris(mlp2)

	mnisttensor.LoadDataset()
	mlp3 := mnisttensor.TrainMNIST(300, 1500, 0.00001)
	mnisttensor.TestMNIST(mlp3)

	pprof.StopCPUProfile()

}
