package main

import (
	"flag"
	"gograd/ng"
	"gograd/trainsets/ngram"
	"log"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"
	"runtime/pprof"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/netlib/blas/netlib"
)

var memprofile = flag.String("memprofile", "", "write memory profile to file")

func main() {
	flag.Parse()

	blas64.Use(netlib.Implementation{})
	ng.BlasImpl = blas64.Implementation()

	go func() {
		http.ListenAndServe("localhost:8080", nil)
	}()

	rand.Seed(1337)

	// a := ng.Tensor1D([]float64{1, 2, 3}) // Shape [3]
	// b := a.Broadcast(2, 3)               // Broadcast to shape [2,3]
	// c := b.Mul(ng.TensorRand(2, 3))      // Perform operation
	// c.Backward()                         // Compute gradients

	// tracers.TraceTensor(c)

	// a.Grad now contains sums along the broadcasted dimensions

	// ft, _ := os.Create("trace.out")
	// trace.Start(ft)
	// defer trace.Stop()

	f, _ := os.Create("cpu.prof")
	defer f.Close() // error handling omitted for example
	if err := pprof.StartCPUProfile(f); err != nil {
		log.Fatal("could not start CPU profile: ", err)
	}

	// iristensor.LoadAndSplitDataset()
	// mlp2 := iristensor.TrainIris(5000, 100, 0.01)
	// iristensor.TestIris(mlp2)

	// mnisttensor.LoadDataset()
	// mlp3 := mnisttensor.TrainMNIST(300, 1500, 0.00001)
	// mnisttensor.TestMNIST(mlp3)

	// bigram.LoadDataset()
	// mlp := bigram.TrainBigram(50, 32000, 0.1)
	// for range 10 {
	// 	bigram.Predict(mlp, "#")
	// }

	ngram.LoadDataset(3)
	mlp := ngram.TrainNgram(3, 30000, 32, 0.001)
	for range 10 {
		ngram.Predict(mlp, 3)
	}

	// seq_len := 128
	// shakespeare.LoadDataset(seq_len)
	// mlp := shakespeare.TrainShakespeare(seq_len, 500, 512, 0.001)
	// for range 10 {
	// 	shakespeare.Predict(mlp, seq_len)
	// }

	pprof.StopCPUProfile()

}
