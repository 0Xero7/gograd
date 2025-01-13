package main

import (
	"flag"
	"fmt"
	"gograd/trainsets/iris"
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

148 out of 150 correct.  98.66666666666667
Train accuracy: [99] of [100] = 0.99
Test accuracy: [49] of [50] = 0.98
# params: 291
Alloc = 1 MB
Total Alloc = 9926 MB


AFTER:
*/

func main() {
	flag.Parse()

	rand.Seed(1337)

	iris.LoadAndSplitDataset()
	printMemStats()
	mlp := iris.TrainIris(1000, 100, 0.01)
	printMemStats()
	iris.TestIris(mlp)
	printMemStats()

	// mnist.LoadDataset()
	// mlp2 := mnist.TrainMNIST(100, 1300, 0.5)
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

	// 	fmt.Println(">", mlp.Predict(dataValues))
	// }
}
