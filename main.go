package main

import (
	"flag"
	"fmt"
	"gograd/trainsets/mnist"
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

======================= 1000 epochs =============================================
148 out of 150 correct.  98.66666666666667
Train accuracy: [99] of [100] = 0.99
Test accuracy: [49] of [50] = 0.98
# params: 291
Alloc = 1 MB
Total Alloc = 9926 MB

======================= 100 epochs =============================================
131 out of 150 correct.  87.33333333333333
Train accuracy: [89] of [100] = 0.89
Test accuracy: [42] of [50] = 0.84
# params: 291
Alloc = 595 MB
Total Alloc = 1015 MB

AFTER:
*/

/*
BEFORE POOLING:
Params:
Seed = 1337
Train = 1000, 100, 0.01

144 out of 150 correct.  96
Train accuracy: [97] of [100] = 0.97
Test accuracy: [47] of [50] = 0.94
# params: 291
Alloc = 8778 MB
Total Alloc = 11745 MB

*/

func main() {
	flag.Parse()

	rand.Seed(1337)

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

	mnist.LoadDataset()
	mlp2 := mnist.TrainMNIST(50, 64, 0.000001)
	mnist.TestMNIST(mlp2)

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
