package mnisttensor

import (
	"bytes"
	"fmt"
	"image/jpeg"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"time"

	"gograd/ng"
	"gograd/nn"
	"gograd/nn/initializers"
	lossfunctions "gograd/nn/loss_functions"
	tensorlayers "gograd/nn/tensor_layers"
	"gograd/optimizers"
)

func printMemStats() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	bToMb := func(b uint64) uint64 {
		return b / 1024 / 1024
	}

	fmt.Printf("Alloc = %v MB\n", bToMb(m.Alloc))
	fmt.Printf("Total Alloc = %v MB\n", bToMb(m.TotalAlloc))
}

var trainInputs [][]float64
var trainOutputs [][]float64

var testInputs [][]float64
var testOutputs [][]float64

func LoadDataset() {
	allInputs := make([][]float64, 0)
	allOutputs := make([][]float64, 0)

	fmt.Println("Loading dataset...")
	for num := range 10 {
		p := filepath.Join("trainingSet", fmt.Sprint(num))
		entries, _ := os.ReadDir(p)
		for _, entry := range entries {
			// if ww > 5 {
			// 	break
			// }

			data, _ := os.ReadFile(filepath.Join(p, entry.Name()))
			image, _ := jpeg.Decode(bytes.NewReader(data))
			dataValues := make([]float64, 0)
			for y := range 28 {
				for x := range 28 {
					r, _, _, _ := image.At(x, y).RGBA()
					dataValues = append(dataValues, float64(r))
				}
			}
			allInputs = append(allInputs, dataValues)
			allOutputs = append(allOutputs, ng.OneHot(num, 10))
		}
	}

	rand.Shuffle(len(allInputs), func(i, j int) {
		allInputs[i], allInputs[j] = allInputs[j], allInputs[i]
		allOutputs[i], allOutputs[j] = allOutputs[j], allOutputs[i]
	})

	trainInputs, trainOutputs = allInputs[0:32000], allOutputs[0:32000]
	testInputs, testOutputs = allInputs[32000:], allOutputs[32000:]

	fmt.Printf("Dataset loaded. Number of files = %d. Train set = %d, Test set = %d\n", len(allInputs), len(trainInputs), len(testInputs))
	printMemStats()
}

func TrainMNIST(iterations, batchSize int, learningRate float64) *nn.MLPTensor {
	mlp := nn.NewMLPTensor(784, []nn.TensorLayer{
		tensorlayers.Linear(784, 512, &initializers.HeInitializer{}),
		tensorlayers.ReLu(512),

		tensorlayers.Linear(512, 256, &initializers.HeInitializer{}),
		tensorlayers.ReLu(256),

		tensorlayers.Linear(256, 128, &initializers.HeInitializer{}),
		tensorlayers.ReLu(128),

		tensorlayers.Linear(128, 10, &initializers.XavierInitializer{}),
	})
	// ng.TTensorPool.Mark()

	params := mlp.Parameters()
	fmt.Println("Created MLP")
	printMemStats()

	adam := optimizers.NewAdamTensor(learningRate)

	totalTime := 0.0

	for epoch := range iterations {
		batch := make([]int, 0)

		xsf := make([][]float64, batchSize)
		ysf := make([][]float64, batchSize)

		batchIndex := 0
		for len(batch) < batchSize {
			r := rand.Intn(len(trainInputs))
			if slices.Contains(batch, r) {
				continue
			}
			batch = append(batch, r)

			xsf[batchIndex] = trainInputs[r]
			ysf[batchIndex] = trainOutputs[r]
			batchIndex++
		}

		// Forward Pass
		fpStart := time.Now().UnixMilli()

		// probabilities := make([]*ng.Tensor, batchSize)
		// classes := make([]int, batchSize)

		xs := ng.Tensor2D(xsf)
		ys := ng.Tensor2D(ysf)

		// for index := range batchSize {
		// probabilities[index] = mlp.Call(trainInputs[batch[index]])
		// classes[index] = int(trainOutputs[batch[index]])
		// }

		logits := mlp.Call(xs)

		fmt.Println(logits.Shape, ys.Shape)

		loss := lossfunctions.TensorCrossEntropyProbDist(logits, ys)

		fpEnd := time.Now().UnixMilli()
		fmt.Printf("Epoch = %d, Loss = %.12f\n", epoch, loss.Value)
		fmt.Printf("Forward Pass Time = %f\n", float64(fpEnd-fpStart)/1000.0)

		// Backward Pass
		bpStart := time.Now().UnixMilli()
		loss.Backward()
		bpEnd := time.Now().UnixMilli()
		fmt.Printf("Backward Pass Time = %f\n", float64(bpEnd-bpStart)/1000)

		// if epoch == 30 {
		// 	tracers.TraceTensor2(loss, epoch)
		// }

		// Update
		upStart := time.Now().UnixMilli()
		adam.Step(params)
		upEnd := time.Now().UnixMilli()
		fmt.Printf("Update Time = %f\n", float64(upEnd-upStart)/1000)

		// loss.Clear()

		totalTime += float64((upEnd + bpEnd + fpEnd) - (upStart + bpStart + fpStart))
		// fmt.Println()
		// fmt.Println(ng.TValuePool.GetCapacity())
		// printMemStats()
		// fmt.Println()
		fmt.Printf("Epoch %d completed in %f. [%f per epoch].\n\n", epoch, float64((upEnd+bpEnd+fpEnd)-(upStart+bpStart+fpStart))/1000, totalTime/float64(epoch+1))

	}

	return mlp
}

func argmax(vals []float64) int {
	best := -1
	bestVal := -100000000000.0

	for i := range vals {
		if vals[i] > bestVal {
			bestVal = vals[i]
			best = i
		}
	}

	return best
}

func TestMNIST(mlp *nn.MLPTensor) {
	accuracy := 0
	trainAccuracy := 0
	testAccuracy := 0
	total := len(trainInputs) + len(testInputs)
	for i := range len(trainInputs) {
		// ng.TTensorPool.ClearUptoMark()

		class := mlp.Call(ng.Tensor2D([][]float64{trainInputs[i]}))
		if argmax(class.Value) == argmax(trainOutputs[i]) {
			accuracy++
			trainAccuracy++
		}

		if (1+i)%100 == 0 {
			fmt.Println(accuracy, "out of", (1 + i), "correct. ", (float64(accuracy)*100.0)/float64((1+i)))
			// fmt.Printf("Train accuracy: [%d] of [%d] = %.2f\n", trainAccuracy, len(trainInputs), float64(trainAccuracy)/float64(len(trainInputs)))
		}
	}
	for i := range len(testInputs) {
		class := mlp.Call(ng.Tensor2D([][]float64{testInputs[i]}))
		if argmax(class.Value) == argmax(testOutputs[i]) {
			accuracy++
			testAccuracy++
		}

		if (1+i)%100 == 0 {
			fmt.Println(testAccuracy, "out of", (1 + i), "correct. ", (float64(accuracy)*100.0)/float64(len(trainInputs)+(1+i)))
		}
	}

	fmt.Println(accuracy, "out of", total, "correct. ", (float64(accuracy)*100.0)/float64(total))
	fmt.Printf("Train accuracy: [%d] of [%d] = %.2f\n", trainAccuracy, len(trainInputs), float64(trainAccuracy)/float64(len(trainInputs)))
	fmt.Printf("Test accuracy: [%d] of [%d] = %.2f\n", testAccuracy, len(testInputs), float64(testAccuracy)/float64(len(testInputs)))
	fmt.Println("# params:", mlp.ParameterCount())
}
