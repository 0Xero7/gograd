package iristensor

import (
	"fmt"
	"gograd/ng"
	"gograd/nn"
	"gograd/nn/initializers"
	lossfunctions "gograd/nn/loss_functions"
	tensorlayers "gograd/nn/tensor_layers"
	"gograd/optimizers"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"
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

var trainInputs *ng.Tensor
var trainOutputs *ng.Tensor

var testInputs []*ng.Tensor
var testOutputs []*ng.Tensor

func LoadAndSplitDataset() {
	data, err := os.ReadFile("iris.csv")
	if err != nil {
		panic(err)
	}
	dataStr := string(data)
	lines := strings.Split(dataStr, "\n")
	for i := range len(lines) {
		lines[i] = strings.TrimSpace(lines[i])
	}
	rand.Shuffle(len(lines), func(i, j int) {
		lines[i], lines[j] = lines[j], lines[i]
	})

	split := 100

	trainSplit := lines[0:split]
	testSplit := lines[split:150]

	// trainInputs = make([]*ng.Tensor, 0)
	// trainOutputs = make([]int, 0)
	inputX := make([]float64, 0)
	outputX := make([]float64, 0)
	for _, line := range trainSplit {
		d := strings.Split(line, ",")
		x1, _ := strconv.ParseFloat(d[0], 64)
		x2, _ := strconv.ParseFloat(d[1], 64)
		x3, _ := strconv.ParseFloat(d[2], 64)
		x4, _ := strconv.ParseFloat(d[3], 64)
		y, _ := strconv.ParseFloat(d[4], 64)

		inputX = append(inputX, x1, x2, x3, x4)
		yy := []float64{0, 0, 0}
		yy[int(y)] = 1.0
		outputX = append(outputX, yy...)

		// trainInputs = append(trainInputs, ng.NewTensorFlat([]float64{x1, x2, x3, x4}, []int{4}))
		// trainOutputs = append(trainOutputs, int(y))
	}
	trainInputs = ng.NewTensorFlat(inputX, []int{split, 4})
	trainOutputs = ng.NewTensorFlat(outputX, []int{split, 3})
	fmt.Println(trainInputs.Shape, trainOutputs.Shape)

	testInputs = []*ng.Tensor{}
	testOutputs = []*ng.Tensor{}
	for _, line := range testSplit {
		d := strings.Split(line, ",")
		x1, _ := strconv.ParseFloat(d[0], 64)
		x2, _ := strconv.ParseFloat(d[1], 64)
		x3, _ := strconv.ParseFloat(d[2], 64)
		x4, _ := strconv.ParseFloat(d[3], 64)
		y, _ := strconv.ParseFloat(d[4], 64)

		testInputs = append(testInputs, ng.NewTensorFlat([]float64{x1, x2, x3, x4}, []int{4}))
		yy := []float64{0, 0, 0}
		yy[int(y)] = 1.0
		testOutputs = append(testOutputs, ng.NewTensorFlat([]float64{yy[0], yy[1], yy[2]}, []int{3}))
	}
}

func TrainIris(iterations, batchSize int, learningRate float64) *nn.MLPTensor {
	mlp := nn.NewMLPTensor(4, []nn.TensorLayer{
		tensorlayers.Linear(4, 32, &initializers.HeInitializer{}),
		tensorlayers.ReLu(32),

		tensorlayers.Linear(32, 16, &initializers.HeInitializer{}),
		tensorlayers.ReLu(16),

		tensorlayers.Linear(16, 3, &initializers.HeInitializer{}),
		tensorlayers.SoftMax(3),
	})

	ng.TTensorPool.Mark()

	optimizer := optimizers.NewAdamTensor(learningRate)

	totalTime := 0.0

	for epoch := range iterations {
		// batch := make([]int, 0)
		// for len(batch) < batchSize {
		// 	r := rand.Intn(len(trainInputs))
		// 	if slices.Contains(batch, r) {
		// 		continue
		// 	}
		// 	batch = append(batch, r)
		// }

		// Forward Pass
		fpStart := time.Now().UnixMilli()

		// results := make([]*ng.Tensor, batchSize)
		// ys := make([]int, batchSize)

		// for index := range batchSize {
		output := mlp.Call(trainInputs)
		// fmt.Println("~~~", output.Shape)
		// results[index] = output
		// ys[index] = int(trainOutputs[index])
		// target := int(trainOutputs[index])
		// }

		loss := lossfunctions.CrossEntropyLoss(output, trainOutputs)

		fpEnd := time.Now().UnixMilli()
		// fmt.Printf("Epoch = %d, Loss = %.12f\n", epoch, loss.Value)
		// fmt.Printf("Forward Pass Time = %f\n", float64(fpEnd-fpStart)/1000.0)

		// Backward Pass
		bpStart := time.Now().UnixMilli()
		// for _, param := range params {
		// 	param.Grad = 0
		// }
		loss.Backward()
		bpEnd := time.Now().UnixMilli()
		// fmt.Printf("Backward Pass Time = %f\n", float64(bpEnd-bpStart)/1000)

		// Update
		upStart := time.Now().UnixMilli()
		params := ng.PathT
		optimizer.Step(params)
		for j := range params {
			for k := range params[j].Value {
				params[j].Value[k] -= params[j].Grad[k] * learningRate
			}
		}
		upEnd := time.Now().UnixMilli()
		// fmt.Printf("Update Time = %f\n", float64(upEnd-upStart)/1000)

		// loss.Clear()

		totalTime += float64((upEnd + bpEnd + fpEnd) - (upStart + bpStart + fpStart))
		if epoch%100 == 0 {
			// fmt.Println(ng.TValuePool.GetCapacity())
			printMemStats()
			fmt.Printf("Epoch %d completed in %f. [%f per epoch] Loss = %.4f.\n\n", epoch, float64((upEnd+bpEnd+fpEnd)-(upStart+bpStart+fpStart))/1000, totalTime/float64(epoch+1), loss.Value)
		}

		// if epoch == iterations-1 {
		// 	tracers.TraceTensor2(loss, epoch)
		// 	fmt.Println("UwU")
		// }
		// fmt.Println(ng.IdGen.Load())
		loss = nil
		ng.TTensorPool.Reset()
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

func TestIris(mlp *nn.MLPTensor) {
	accuracy := 0
	// trainAccuracy := 0
	testAccuracy := 0
	// total := 150
	// for i := range len(trainInputs) {
	// 	class := mlp.Predict(trainInputs[i])
	// 	fmt.Printf("Output #%d: Actual: %d Got: %d\n", i, trainOutputs[i], class)
	// 	if class == int(trainOutputs[i]) {
	// 		accuracy++
	// 		trainAccuracy++
	// 	}
	// }
	for i := range len(testInputs) {
		ng.TTensorPool.ClearUptoMark()

		class := mlp.Predict(testInputs[i])
		fmt.Printf("Output #%d: Actual: %v Got: %d\n", i, argmax(testOutputs[i].Value), class)
		if class == argmax(testOutputs[i].Value) {
			accuracy++
			testAccuracy++
		}
		fmt.Printf("Test accuracy: [%d] of [%d] = %.2f\n", testAccuracy, len(testInputs), float64(testAccuracy)/float64(len(testInputs)))
	}

	// fmt.Println(accuracy, "out of", total, "correct. ", (float64(accuracy)*100.0)/float64(total))
	// fmt.Printf("Train accuracy: [%d] of [%d] = %.2f\n", trainAccuracy, trainInputs.Shape[0], float64(trainAccuracy)/float64(trainInputs.Shape[0]))
	// fmt.Println("# params:", len(mlp.Parameters()))
}
