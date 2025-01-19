package main

import (
	"flag"
	iristensor "gograd/trainsets/iris_tensor"
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

	// inputs := ng.NewTensorFlat([]float64{
	// 	2, -5, 7,
	// 	-4, 3, 9,
	// }, []int{2, 3}).SoftMax(1)
	// fmt.Println(inputs)

	// target := ng.NewTensorFlat([]float64{
	// 	0.2, 0.3, 0.5,
	// 	0.1, 0.85, 0.05,
	// }, []int{2, 3})
	// fmt.Println(target)
	// inputs := ng.NewTensorFlat([]float64{
	// 	2, -5, 3,
	// 	-7, 4, 9,
	// }, []int{2, 3})
	// sf := inputs.SoftMax(1)
	// init := *ng.NewTensorFlat([]float64{-0.5, 1.44, 3, -2, 5, 0}, []int{2, 3})
	// sf.Backward(init)

	// fmt.Println(inputs)
	// fmt.Println(sf)
	// tracers.TraceTensor(sf)

	// target := ng.NewTensorFlat([]float64{
	// 	0.2, 0.8,
	// }, []int{1, 2})

	// loss := lossfunctions.CrossEntropyLoss(sf, target)
	// loss.Backward()
	// fmt.Println(inputs)
	// fmt.Println(target)
	// fmt.Println(loss)

	// tracers.TraceTensor(loss)

	// a := ng.Tensor2D([][]float64{
	// 	{1, 2},
	// 	{3, 4},
	// 	{5, 6},
	// })
	// b := ng.Tensor2D([][]float64{
	// 	{6, 7, 8},
	// 	{9, 10, 11},
	// })
	// c := a.MatMul(b)
	// d := ng.Tensor2D(([][]float64{
	// 	{-5, 6, 1},
	// 	{2, -4, 0},
	// 	{7, -9, -3},
	// }))
	// e := c.Add(d)
	// e.Backward()
	// fmt.Println(e)

	// fmt.Println(a)
	// fmt.Println(b)

	// layer := tensorlayers.Linear(2, 3, &initializers.SimpleInitializer{})
	// out := layer.Call(ng.Tensor2D([][]float64{
	// 	{1, 2},
	// 	{3, 4},
	// }))
	// fmt.Println(out)
	// fmt.Println(out.Shape)

	f, _ := os.Create("cpu.prof")
	defer f.Close() // error handling omitted for example
	if err := pprof.StartCPUProfile(f); err != nil {
		log.Fatal("could not start CPU profile: ", err)
	}

	iristensor.LoadAndSplitDataset()
	mlp2 := iristensor.TrainIris(5000, 100, 0.01)
	iristensor.TestIris(mlp2)

	// mnisttensor.LoadDataset()

	// mlp3 := mnisttensor.TrainMNIST(50, 512, 0.00001)
	pprof.StopCPUProfile()

	// mnisttensor.TestMNIST(mlp3)
}
