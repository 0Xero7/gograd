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

var trainInputs []*ng.Tensor
var trainOutputs []int

var testInputs []*ng.Tensor
var testOutputs []float64

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

	trainSplit := lines[0:100]
	testSplit := lines[100:150]

	trainInputs = make([]*ng.Tensor, 0)
	trainOutputs = make([]int, 0)
	for _, line := range trainSplit {
		d := strings.Split(line, ",")
		x1, _ := strconv.ParseFloat(d[0], 64)
		x2, _ := strconv.ParseFloat(d[1], 64)
		x3, _ := strconv.ParseFloat(d[2], 64)
		x4, _ := strconv.ParseFloat(d[3], 64)
		y, _ := strconv.ParseFloat(d[4], 64)

		trainInputs = append(trainInputs, ng.NewTensorFlat([]float64{x1, x2, x3, x4}, []int{4}))
		trainOutputs = append(trainOutputs, int(y))
	}
	fmt.Println(trainOutputs)

	testInputs = []*ng.Tensor{}
	testOutputs = []float64{}
	for _, line := range testSplit {
		d := strings.Split(line, ",")
		x1, _ := strconv.ParseFloat(d[0], 64)
		x2, _ := strconv.ParseFloat(d[1], 64)
		x3, _ := strconv.ParseFloat(d[2], 64)
		x4, _ := strconv.ParseFloat(d[3], 64)
		y, _ := strconv.ParseFloat(d[4], 64)

		testInputs = append(testInputs, ng.NewTensorFlat([]float64{x1, x2, x3, x4}, []int{4}))
		testOutputs = append(testOutputs, y)
	}
}

func TrainIris(iterations, batchSize int, learningRate float64) *nn.MLPTensor {
	mlp := nn.NewMLPTensor(4, []nn.TensorLayer{
		tensorlayers.Linear(4, 32, &initializers.HeInitializer{}),
		tensorlayers.Tanh(32),

		tensorlayers.Linear(32, 16, &initializers.HeInitializer{}),
		tensorlayers.Tanh(16),

		tensorlayers.Linear(16, 3, &initializers.HeInitializer{}),
		tensorlayers.SoftMax(3),
	})

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

		results := make([]*ng.Tensor, batchSize)
		ys := make([]int, batchSize)

		for index := range batchSize {
			output := mlp.Call(trainInputs[index])
			results[index] = output
			ys[index] = int(trainOutputs[index])
			// target := int(trainOutputs[index])
		}

		loss := lossfunctions.BatchCrossEntropyTensor(results, ys)

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
		// for j := range params {
		// 	for k := range params[j].Value {
		// 		params[j].Value[k] -= params[j].Grad[k] * learningRate
		// 	}
		// }
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
		// 	tracers.TraceTensor(loss)
		// }

		// fmt.Println(ng.IdGen.Load())
		loss = nil
		ng.TValuePool.Reset()
	}

	return mlp
}

func TestIris(mlp *nn.MLPTensor) {
	accuracy := 0
	trainAccuracy := 0
	testAccuracy := 0
	total := len(trainInputs) + len(testInputs)
	for i := range len(trainInputs) {
		class := mlp.Predict(trainInputs[i])
		fmt.Printf("Output #%d: Actual: %.0f Got: %d\n", i, trainOutputs[i], class)
		if class == int(trainOutputs[i]) {
			accuracy++
			trainAccuracy++
		}
	}
	for i := range len(testInputs) {
		class := mlp.Predict(testInputs[i])
		fmt.Printf("Output #%d: Actual: %.0f Got: %d\n", i, testOutputs[i], class)
		if class == int(testOutputs[i]) {
			accuracy++
			testAccuracy++
		}
	}

	fmt.Println(accuracy, "out of", total, "correct. ", (float64(accuracy)*100.0)/float64(total))
	fmt.Printf("Train accuracy: [%d] of [%d] = %.2f\n", trainAccuracy, len(trainInputs), float64(trainAccuracy)/float64(len(trainInputs)))
	fmt.Printf("Test accuracy: [%d] of [%d] = %.2f\n", testAccuracy, len(testInputs), float64(testAccuracy)/float64(len(testInputs)))
	fmt.Println("# params:", len(mlp.Parameters()))
}
