package main

import (
	"fmt"
	"math/rand"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"
)

func main() {
	data, err := os.ReadFile("iris.csv")
	if err != nil {
		panic(err)
	}
	dataStr := string(data)
	lines := strings.Split(dataStr, "\n")
	rand.Shuffle(len(lines), func(i, j int) {
		temp := string(lines[i])
		lines[i] = string(lines[j])
		lines[j] = temp
	})

	trainSplit := lines[0:100]
	testSplit := lines[100:]

	inputs := [][]*Value{}
	outputs := []float64{}
	for _, line := range trainSplit {
		d := strings.Split(line, ",")
		x1, _ := strconv.ParseFloat(d[0], 64)
		x2, _ := strconv.ParseFloat(d[1], 64)
		x3, _ := strconv.ParseFloat(d[2], 64)
		x4, _ := strconv.ParseFloat(d[3], 64)
		y, _ := strconv.ParseFloat(d[4], 64)

		inputs = append(inputs, []*Value{
			NewValueLiteral(x1),
			NewValueLiteral(x2),
			NewValueLiteral(x3),
			NewValueLiteral(x4),
		})

		outputs = append(outputs, y)
	}

	testInputs := [][]*Value{}
	testOutputs := []float64{}
	for _, line := range testSplit {
		d := strings.Split(line, ",")
		x1, _ := strconv.ParseFloat(d[0], 64)
		x2, _ := strconv.ParseFloat(d[1], 64)
		x3, _ := strconv.ParseFloat(d[2], 64)
		x4, _ := strconv.ParseFloat(d[3], 64)
		y, _ := strconv.ParseFloat(d[4], 64)

		testInputs = append(inputs, []*Value{
			NewValueLiteral(x1),
			NewValueLiteral(x2),
			NewValueLiteral(x3),
			NewValueLiteral(x4),
		})

		testOutputs = append(outputs, y)
	}

	// for num := range 10 {
	// 	p := filepath.Join("trainingSet", fmt.Sprint(num))
	// 	entries, _ := os.ReadDir(p)
	// 	for ww, entry := range entries {
	// 		if ww > 5 {
	// 			// break
	// 		}

	// 		data, _ := os.ReadFile(filepath.Join(p, entry.Name()))
	// 		image, _ := jpeg.Decode(bytes.NewReader(data))
	// 		dataValues := []*Value{}
	// 		for y := range 28 {
	// 			for x := range 28 {
	// 				r, _, _, _ := image.At(x, y).RGBA()
	// 				dataValues = append(dataValues, NewValueLiteral(float64(r)))
	// 			}
	// 		}
	// 		inputs = append(inputs, dataValues)
	// 		output := float64(num)
	// 		outputs = append(outputs, output)
	// 	}
	// }

	mlp := NewMLP(4, []*Layer{
		NewLayer(4, 8, Sigmoid),
		NewLayer(8, 3, Linear),
	})

	// mlp := NewMLP(784, []*Layer{
	// 	NewLayer(784, 256, Linear),
	// 	NewLayer(256, 256, Tanh),
	// 	NewLayer(256, 256, Linear),
	// 	NewLayer(256, 256, Tanh),
	// 	NewLayer(256, 10, Linear),
	// })
	params := mlp.Parameters()

	iterations := 1000
	learningRate := 0.02
	batchSize := len(inputs)

	totalTime := 0.0

	for epoch := range iterations {
		batch := make([]int, 0)
		for len(batch) < batchSize {
			r := rand.Intn(len(inputs))
			if slices.Contains(batch, r) {
				continue
			}
			batch = append(batch, r)
		}

		// Forward Pass
		fpStart := time.Now().UnixMilli()
		loss := NewValueLiteral(0)
		for index := range batchSize {
			results := mlp.Call(inputs[index])
			target := outputs[index]

			lossItem := SoftmaxCrossEntropy(results, int(target))
			loss = loss.Add(lossItem)
		}
		loss = loss.Div(NewValueLiteral(float64(batchSize)))
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
	}

	// Get accuracy
	accuracy := 0
	trainAccuracy := 0
	testAccuracy := 0
	for i := range len(trainSplit) {
		class := mlp.Predict(inputs[i])
		if class == int(outputs[i]) {
			accuracy++
			trainAccuracy++
		}
	}
	for i := range len(testSplit) {
		class := mlp.Predict(testInputs[i])
		if class == int(testOutputs[i]) {
			accuracy++
			testAccuracy++
		}
	}

	fmt.Println(accuracy, "out of", len(lines), "correct. ", (float64(accuracy)*100.0)/float64(len(lines)))
	fmt.Println("Train accuracy:", float64(trainAccuracy)/float64(len(trainSplit)))
	fmt.Println("Test accuracy:", float64(testAccuracy)/float64(len(testSplit)))
	fmt.Println("# params:", len(mlp.Parameters()))

	// for {
	// 	reader := bufio.NewReader(os.Stdin)
	// 	line, _ := reader.ReadString('\n')
	// 	line = strings.TrimSpace(line)

	// 	inp := filepath.Join("./testSet", "img_"+line+".jpg")
	// 	fmt.Println(inp)
	// 	data, err := os.ReadFile(inp)
	// 	if err != nil {
	// 		log.Fatal(err)
	// 	}
	// 	image, _ := jpeg.Decode(bytes.NewReader(data))
	// 	dataValues := []*Value{}
	// 	for y := range 28 {
	// 		for x := range 28 {
	// 			r, _, _, _ := image.At(x, y).RGBA()
	// 			dataValues = append(dataValues, NewValueLiteral(float64(r)))
	// 		}
	// 	}

	// 	result := mlp.Predict(dataValues)
	// 	fmt.Println(result)
	// 	// fmt.Println(result[0].Value)
	// }
}

// Get predictions for a single input
func (mlp *MLP) Predict(input []*Value) int {
	// Forward pass
	values := mlp.Call(input)

	// Find max logit (don't need full softmax for prediction)
	maxIdx := 0
	maxVal := values[0].Value
	for i, v := range values {
		if v.Value > maxVal {
			maxVal = v.Value
			maxIdx = i
		}
	}

	return maxIdx
}
