package iris

import (
	"fmt"
	"gograd/ng"
	"gograd/nn"
	"gograd/nn/layers"
	lossfunctions "gograd/nn/loss_functions"
	"math/rand"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"
)

var trainInputs [][]*ng.Value
var trainOutputs []float64

var testInputs [][]*ng.Value
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

	trainInputs = [][]*ng.Value{}
	trainOutputs = []float64{}
	for _, line := range trainSplit {
		d := strings.Split(line, ",")
		x1, _ := strconv.ParseFloat(d[0], 64)
		x2, _ := strconv.ParseFloat(d[1], 64)
		x3, _ := strconv.ParseFloat(d[2], 64)
		x4, _ := strconv.ParseFloat(d[3], 64)
		y, _ := strconv.ParseFloat(d[4], 64)

		trainInputs = append(trainInputs, []*ng.Value{
			ng.NewValueLiteral(x1),
			ng.NewValueLiteral(x2),
			ng.NewValueLiteral(x3),
			ng.NewValueLiteral(x4),
		})

		trainOutputs = append(trainOutputs, y)
	}

	testInputs = [][]*ng.Value{}
	testOutputs = []float64{}
	for _, line := range testSplit {
		d := strings.Split(line, ",")
		x1, _ := strconv.ParseFloat(d[0], 64)
		x2, _ := strconv.ParseFloat(d[1], 64)
		x3, _ := strconv.ParseFloat(d[2], 64)
		x4, _ := strconv.ParseFloat(d[3], 64)
		y, _ := strconv.ParseFloat(d[4], 64)

		testInputs = append(testInputs, []*ng.Value{
			ng.NewValueLiteral(x1),
			ng.NewValueLiteral(x2),
			ng.NewValueLiteral(x3),
			ng.NewValueLiteral(x4),
		})

		testOutputs = append(testOutputs, y)
	}
}

func TrainIris(iterations, batchSize int, learningRate float64) *nn.MLP {
	mlp := nn.NewMLP(4, []nn.Layer{
		layers.Linear(4, 32),
		layers.Tanh(32),
		layers.Linear(32, 3),
	})
	params := mlp.Parameters()

	totalTime := 0.0

	for epoch := range iterations {
		batch := make([]int, 0)
		for len(batch) < batchSize {
			r := rand.Intn(len(trainInputs))
			if slices.Contains(batch, r) {
				continue
			}
			batch = append(batch, r)
		}

		// Forward Pass
		fpStart := time.Now().UnixMilli()
		loss := ng.NewValueLiteral(0)
		for index := range batchSize {
			results := mlp.Call(trainInputs[index])
			target := trainOutputs[index]

			lossItem := lossfunctions.SoftmaxCrossEntropy(results, int(target))
			loss = loss.Add(lossItem)
		}
		loss = loss.Div(ng.NewValueLiteral(float64(batchSize)))
		fpEnd := time.Now().UnixMilli()
		// fmt.Printf("Epoch = %d, Loss = %.12f\n", epoch, loss.Value)
		// fmt.Printf("Forward Pass Time = %f\n", float64(fpEnd-fpStart)/1000.0)

		// Backward Pass
		bpStart := time.Now().UnixMilli()
		for _, param := range params {
			param.Grad = 0
		}
		loss.Backward()
		bpEnd := time.Now().UnixMilli()
		// fmt.Printf("Backward Pass Time = %f\n", float64(bpEnd-bpStart)/1000)

		// Update
		upStart := time.Now().UnixMilli()
		for j := range params {
			params[j].Value -= params[j].Grad * learningRate
		}
		upEnd := time.Now().UnixMilli()
		// fmt.Printf("Update Time = %f\n", float64(upEnd-upStart)/1000)

		// loss.Clear()
		loss = nil

		totalTime += float64((upEnd + bpEnd + fpEnd) - (upStart + bpStart + fpStart))
		if epoch%100 == 0 {
			fmt.Printf("Epoch %d completed in %f. [%f per epoch].\n\n", epoch, float64((upEnd+bpEnd+fpEnd)-(upStart+bpStart+fpStart))/1000, totalTime/float64(epoch+1))
		}

		fmt.Println(ng.IdGen.Load())
	}

	return mlp
}

func TestIris(mlp *nn.MLP) {
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
