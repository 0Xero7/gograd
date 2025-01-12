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
	// inputs = inputs[0:1]

	mlp := NewMLP(841, []int{2}, 1)
	params := mlp.Parameters()
	fmt.Println(len(params))

	// mlp.Call(inputs[0])[0].CalculateBackwardPath()

	iterations := 10
	learningRate := 0.01
	batchSize := 50
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
		fmt.Printf("Forward Pass Time = %d\n", (fpEnd-fpStart)/1000)

		// Backward Pass
		bpStart := time.Now().UnixMilli()
		for _, param := range params {
			param.Grad = 0
		}
		loss.Backward()
		bpEnd := time.Now().UnixMilli()
		fmt.Printf("Backward Pass Time = %d\n", (bpEnd-bpStart)/1000)

		// Update
		upStart := time.Now().UnixMilli()
		for j := range params {
			params[j].Value -= params[j].Grad * learningRate
		}
		upEnd := time.Now().UnixMilli()
		fmt.Printf("Update Time = %d\n", (upEnd-upStart)/1000)
		fmt.Printf("Epoch %d completed in %d.\n\n", epoch, ((upEnd+bpEnd+fpEnd)-(upStart+bpStart+fpStart))/1000)
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
