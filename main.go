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
	outputs := []*Value{}

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
			for y := range 29 {
				for x := range 29 {
					r, _, _, _ := image.At(x, y).RGBA()
					dataValues = append(dataValues, NewValueLiteral(float64(r)))
				}
			}
			inputs = append(inputs, dataValues)
			outputs = append(outputs, NewValueLiteral(float64(num)))
		}
	}

	mlp := NewMLP(841, []int{32}, 1)
	params := mlp.Parameters()

	iterations := 100
	learningRate := 0.01
	batchSize := 50

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
			output := mlp.Call(inputs[index])[0]
			target := outputs[index]

			loss = loss.Add(target.Sub(output).Pow(2))
		}
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

		totalTime += float64((upEnd + bpEnd + fpEnd) - (upStart + bpStart + fpStart))
		fmt.Printf("Epoch %d completed in %f. [%f per epoch].\n\n", epoch, float64((upEnd+bpEnd+fpEnd)-(upStart+bpStart+fpStart))/1000, totalTime/float64(epoch+1))

	}

	for {
		reader := bufio.NewReader(os.Stdin)
		line, _ := reader.ReadString('\n')
		line = strings.TrimSpace(line)

		inp := filepath.Join("./testSet", line)
		fmt.Println(inp)
		data, err := os.ReadFile(inp)
		if err != nil {
			log.Fatal(err)
		}
		image, _ := jpeg.Decode(bytes.NewReader(data))
		dataValues := []*Value{}
		for y := range 29 {
			for x := range 29 {
				r, _, _, _ := image.At(x, y).RGBA()
				dataValues = append(dataValues, NewValueLiteral(float64(r)))
			}
		}

		result := mlp.Call(dataValues)
		fmt.Println(result)
		fmt.Println(result[0].Value)
	}
}
