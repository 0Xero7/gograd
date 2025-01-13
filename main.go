package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"gograd/ng"
	"gograd/trainsets/mnist"
	"image/jpeg"
	"os"
	"path/filepath"
	"runtime"
	"strings"
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

func main() {
	flag.Parse()

	mnist.LoadDataset()
	mlp := mnist.TrainMNIST(50, 100, 0.09)
	// mnist.TestMNIST(mlp)

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Println("Image Id: ")
		img, _ := reader.ReadString('\n')
		s := strings.TrimSpace(img)
		path := filepath.Join("testSet", "img_"+s+".jpg")
		data, _ := os.ReadFile(path)
		image, _ := jpeg.Decode(bytes.NewReader(data))
		dataValues := []*ng.Value{}
		for y := range 28 {
			for x := range 28 {
				r, _, _, _ := image.At(x, y).RGBA()
				dataValues = append(dataValues, ng.NewValueLiteral(float64(r)))
			}
		}

		fmt.Println(">", mlp.Predict(dataValues))
	}
}
