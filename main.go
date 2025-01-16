package main

import (
	"flag"
	"fmt"
	"gograd/ng"
	"gograd/nn"
	"gograd/nn/initializers"
	tensorlayers "gograd/nn/tensor_layers"
	"math/rand"
	"runtime"
)

var memprofile = flag.String("memprofile", "", "write memory profile to file")

func printMemStats() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	bToMb := func(b uint64) uint64 {
		return b / 1024 / 1024
	}

	fmt.Printf("Alloc = %v MB\n", bToMb(m.Alloc))
	fmt.Printf("Total Alloc = %v MB\n", bToMb(m.TotalAlloc))
}

/*
BEFORE:
train params: 200, 64, 0.0001
nn: mlp := nn.NewMLP(784, []nn.Layer{
		layers.Linear(784, 128, &initializers.HeInitializer{}),
		layers.ReLu(128),

		layers.Linear(128, 64, &initializers.HeInitializer{}),
		layers.ReLu(64),

		layers.Linear(64, 32, &initializers.HeInitializer{}),
		layers.ReLu(32),

		layers.Linear(32, 10, &initializers.SimpleInitializer{}),
	})

Epoch = 199, Loss = 0.000194531579
Forward Pass Time = 0.487000
Backward Pass Time = 0.187000
Update Time = 0.021000
Epoch 199 completed in 0.695000. [871.430000 per epoch].

[1944 out of 3500 correct.  55.542857142857144]
*/

func main() {
	flag.Parse()

	// input := ng.Tensor1D([]float64{1, 2, 3})

	// mlp := nn.NewMLPTensor(3, []nn.TensorLayer{
	// 	tensorlayers.Linear(3, 4, &initializers.HeInitializer{}),
	// 	tensorlayers.ReLu(4),

	// 	tensorlayers.Linear(4, 5, &initializers.XavierInitializer{}),
	// 	tensorlayers.SoftMax(5),
	// })

	// out := mlp.Call(input)
	// loss := lossfunctions.CrossEntropyTensor(out, 3)
	// loss.Backward()

	// traceTensor(loss)

	// fmt.Println("out=", loss)

	// for _, v := range ng.PathT {
	// 	fmt.Println(v)
	// }

	xs := []*ng.Tensor{
		ng.Tensor1D([]float64{2, 3, -1}),
		ng.Tensor1D([]float64{3, -1, 0.5}),
		ng.Tensor1D([]float64{0.5, 1, 1}),
		ng.Tensor1D([]float64{1, 1, -1}),
	}
	ys := []*ng.Tensor{
		ng.Scalar(1),
		ng.Scalar(-1),
		ng.Scalar(-1),
		ng.Scalar(1),
	}

	rand.Seed(1337)

	mlp := nn.NewMLPTensor(3, []nn.TensorLayer{
		tensorlayers.Linear(3, 4, &initializers.SimpleInitializer{}),
		tensorlayers.Linear(4, 4, &initializers.SimpleInitializer{}),
		tensorlayers.Linear(4, 1, &initializers.SimpleInitializer{}),
	})

	for range 100 {
		results := make([]*ng.Tensor, 0)
		for i := range xs {
			results = append(results, mlp.Call(xs[i]))
		}

		loss := ng.Scalar(0)
		for i := range results {
			loss = loss.Add((results[i].Sub(ys[i])).Pow(2))
		}
		loss.Backward()

		fmt.Println(loss)

		for _, p := range mlp.Parameters() {
			for j := range p.Grad {
				p.Value[j] -= p.Grad[j] * 0.01
			}
		}
	}

	// iristensor.LoadAndSplitDataset()
	// mlp := iristensor.TrainIris(1000, 1, 0.0000000001)
	// iristensor.TestIris(mlp)

	// pool := ng.NewValuePool[int](func(index int) *int {
	// 	temp := 1337
	// 	return &temp
	// })

	// for i := range 10 {
	// 	v := pool.Get()
	// 	*v = i
	// 	fmt.Println(*v)
	// }

	// pool.Reset()
	// for range 10 {
	// 	v := pool.Get()
	// 	fmt.Println(*v)
	// }

	// inputs := []*ng.Value{ng.NewValueLiteral(10)}

	// mlp := nn.NewMLP(1, []nn.Layer{
	// 	layers.Linear(1, 2),
	// 	layers.Linear(2, 1),
	// })
	// out := mlp.Call(inputs)
	// trace(out[0])

	// iris.LoadAndSplitDataset()
	// printMemStats()
	// mlp := iris.TrainIris(1000, 100, 0.01)
	// if *memprofile !=
	// 	"" {
	// 	f, err := os.Create(*memprofile)
	// 	if err != nil {
	// 		log.Fatal(err)
	// 	}
	// 	pprof.WriteHeapProfile(f)
	// 	f.Close()
	// }
	// // mlp := iris.TrainIris(1000, 100, 0.01)
	// printMemStats()
	// iris.TestIris(mlp)
	// printMemStats()

	// mnist.LoadDataset()
	// mlp2 := mnist.TrainMNIST(200, 2, 0.0001)
	// mnist.TestMNIST(mlp2)

	// reader := bufio.NewReader(os.Stdin)
	// for {
	// 	fmt.Println("Image Id: ")
	// 	img, _ := reader.ReadString('\n')
	// 	s := strings.TrimSpace(img)
	// 	path := filepath.Join("testSet", "img_"+s+".jpg")
	// 	data, _ := os.ReadFile(path)
	// 	image, _ := jpeg.Decode(bytes.NewReader(data))
	// 	dataValues := []*ng.Value{}
	// 	for y := range 28 {
	// 		for x := range 28 {
	// 			r, _, _, _ := image.At(x, y).RGBA()
	// 			dataValues = append(dataValues, ng.NewValueLiteral(float64(r)))
	// 		}
	// 	}

	// 	fmt.Println(">", mlp2.Predict(dataValues))
	// }
}
