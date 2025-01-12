package main

import (
	"bufio"
	"bytes"
	"fmt"
	"image/jpeg"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"
)

func main() {

	// x1 := NewValueLiteral(2.0)
	// x1.Label = "x1"
	// x2 := NewValueLiteral(0.0)
	// x2.Label = "x2"

	// w1 := NewValueLiteral(-3.0)
	// w1.Label = "w1"
	// w2 := NewValueLiteral(1.0)
	// w2.Label = "w2"

	// b := NewValueLiteral(6.88137535870195432)
	// b.Label = "b"

	// x1w1 := x1.Mul(w1)
	// x1w1.Label = "x1w1"
	// x2w2 := x2.Mul(w2)
	// x2w2.Label = "x2w2"

	// x1w1x2w1 := x1w1.Add(x2w2)
	// x1w1x2w1.Label = "x1w1x2w1"
	// n := x1w1x2w1.Add(b)
	// n.Label = "n"

	// // o := n.Tanh()

	// e := n.Mul(NewValueLiteral(2.0)).Exp()
	// e.Label = "e"
	// o := (e.Sub(NewValueLiteral(1))).Div(e.Add(NewValueLiteral(1)))
	// o.Label = "o"

	// o.PreCalculateBackwardPath()
	// o.Backward()
	// trace(o)

	inputs := [][]*Value{}
	outputs := []float64{}

	for num := range 10 {
		p := filepath.Join("trainingSet", fmt.Sprint(num))
		entries, _ := os.ReadDir(p)
		for ww, entry := range entries {
			if ww > 5 {
				// break
			}

			data, _ := os.ReadFile(filepath.Join(p, entry.Name()))
			image, _ := jpeg.Decode(bytes.NewReader(data))
			dataValues := []*Value{}
			for y := range 28 {
				for x := range 28 {
					r, _, _, _ := image.At(x, y).RGBA()
					dataValues = append(dataValues, NewValueLiteral(float64(r)))
				}
			}
			inputs = append(inputs, dataValues)
			output := float64(num)
			outputs = append(outputs, output)
		}
	}

	mlp := NewMLP(784, []*Layer{
		NewLayer(784, 256, Linear),
		NewLayer(256, 256, Tanh),
		NewLayer(256, 256, Linear),
		NewLayer(256, 256, Tanh),
		NewLayer(256, 10, Linear),
	})
	params := mlp.Parameters()

	iterations := 30
	learningRate := 0.09
	batchSize := 32

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

	for {
		reader := bufio.NewReader(os.Stdin)
		line, _ := reader.ReadString('\n')
		line = strings.TrimSpace(line)

		inp := filepath.Join("./testSet", "img_"+line+".jpg")
		fmt.Println(inp)
		data, err := os.ReadFile(inp)
		if err != nil {
			log.Fatal(err)
		}
		image, _ := jpeg.Decode(bytes.NewReader(data))
		dataValues := []*Value{}
		for y := range 28 {
			for x := range 28 {
				r, _, _, _ := image.At(x, y).RGBA()
				dataValues = append(dataValues, NewValueLiteral(float64(r)))
			}
		}

		result := mlp.Predict(dataValues)
		fmt.Println(result)
		// fmt.Println(result[0].Value)
	}
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
