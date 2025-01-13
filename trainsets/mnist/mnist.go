package mnist

import (
	"bytes"
	"fmt"
	"image/jpeg"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"time"

	"gograd/ng"
	"gograd/nn"
	"gograd/nn/layers"
	lossfunctions "gograd/nn/loss_functions"
)

var trainInputs [][]*ng.Value
var trainOutputs []float64

var testInputs [][]*ng.Value
var testOutputs []float64

func LoadDataset() {
	fmt.Println("Loading dataset...")
	for num := range 10 {
		p := filepath.Join("trainingSet", fmt.Sprint(num))
		entries, _ := os.ReadDir(p)
		for _, entry := range entries {
			// if ww > 5 {
			// break
			// }

			data, _ := os.ReadFile(filepath.Join(p, entry.Name()))
			image, _ := jpeg.Decode(bytes.NewReader(data))
			dataValues := []*ng.Value{}
			for y := range 28 {
				for x := range 28 {
					r, _, _, _ := image.At(x, y).RGBA()
					dataValues = append(dataValues, ng.NewValueLiteral(float64(r)))
				}
			}
			trainInputs = append(trainInputs, dataValues)
			output := float64(num)
			trainOutputs = append(trainOutputs, output)
		}
	}

	rand.Shuffle(len(trainInputs), func(i, j int) {
		trainInputs[i], trainInputs[j] = trainInputs[j], trainInputs[i]
		trainOutputs[i], trainOutputs[j] = trainOutputs[j], trainOutputs[i]
	})

	fmt.Printf("Dataset loaded. Number of files = %d.\n", len(trainInputs))
}

func TrainMNIST(iterations, batchSize int, learningRate float64) *nn.MLP {
	mlp := nn.NewMLP(784, []nn.Layer{
		layers.Linear(784, 128),
		layers.ReLu(128),

		layers.Linear(128, 64),
		layers.Sigmoid(64),

		layers.Linear(64, 32),
		layers.Sigmoid(32),

		layers.Linear(32, 10),
		// nn.NewLayer(512, 256, nn.ReLu),
		// nn.NewLayer(512, 10, nn.Linear),
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
		fmt.Printf("Epoch = %d, Loss = %.12f\n", epoch, loss.Value)
		fmt.Printf("Forward Pass Time = %f\n", float64(fpEnd-fpStart)/1000.0)

		// Backward Pass
		bpStart := time.Now().UnixMilli()
		for _, param := range params {
			param.Grad = 0
		}
		loss.Backward()
		bpEnd := time.Now().UnixMilli()
		fmt.Printf("Backward Pass Time = %f\n", float64(bpEnd-bpStart)/1000)

		// Update
		upStart := time.Now().UnixMilli()
		for j := range params {
			params[j].Value -= params[j].Grad * learningRate
		}
		upEnd := time.Now().UnixMilli()
		fmt.Printf("Update Time = %f\n", float64(upEnd-upStart)/1000)

		loss.Clear()

		totalTime += float64((upEnd + bpEnd + fpEnd) - (upStart + bpStart + fpStart))
		fmt.Printf("Epoch %d completed in %f. [%f per epoch].\n\n", epoch, float64((upEnd+bpEnd+fpEnd)-(upStart+bpStart+fpStart))/1000, totalTime/float64(epoch+1))

		// if *memprofile != "" {
		// 	f, err := os.Create(*memprofile)
		// 	if err != nil {
		// 		log.Fatal(err)
		// 	}
		// 	pprof.WriteHeapProfile(f)
		// 	f.Close()
		// 	break
		// }
	}

	return mlp
}

func TestMNIST(mlp *nn.MLP) {
	accuracy := 0
	trainAccuracy := 0
	// testAccuracy := 0
	total := len(trainInputs) + len(testInputs)
	for i := range len(trainInputs) {
		class := mlp.Predict(trainInputs[i])
		if class == int(trainOutputs[i]) {
			accuracy++
			trainAccuracy++
		}

		if (1+i)%100 == 0 {
			fmt.Println(accuracy, "out of", (1 + i), "correct. ", (float64(accuracy)*100.0)/float64((1+i)))
			// fmt.Printf("Train accuracy: [%d] of [%d] = %.2f\n", trainAccuracy, len(trainInputs), float64(trainAccuracy)/float64(len(trainInputs)))
		}
	}
	// for i := range len(testInputs) {
	// 	class := mlp.Predict(testInputs[i])
	// 	if class == int(testOutputs[i]) {
	// 		accuracy++
	// 		testAccuracy++
	// 	}
	// }

	fmt.Println(accuracy, "out of", total, "correct. ", (float64(accuracy)*100.0)/float64(total))
	fmt.Printf("Train accuracy: [%d] of [%d] = %.2f\n", trainAccuracy, len(trainInputs), float64(trainAccuracy)/float64(len(trainInputs)))
	// fmt.Printf("Test accuracy: [%d] of [%d] = %.2f\n", testAccuracy, len(testInputs), float64(testAccuracy)/float64(len(testInputs)))
	fmt.Println("# params:", len(mlp.Parameters()))
}
