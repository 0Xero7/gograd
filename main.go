package main

import (
	"bytes"
	"fmt"
	"image/jpeg"
	"os"
	"path/filepath"
)

func main() {
	inputs := [][]*Value{}
	outputs := []*Value{}

	for num := range 10 {
		p := filepath.Join("trainingSet", fmt.Sprint(num))
		entries, _ := os.ReadDir(p)
		for _, entry := range entries {
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

	// layer := NewMLP(3, []int{4, 4}, 1)

	// xs := [][]*Value{
	// 	{NewValueLiteral(2.0), NewValueLiteral(3.0), NewValueLiteral(-1.0)},
	// 	{NewValueLiteral(3.0), NewValueLiteral(-1.0), NewValueLiteral(0.5)},
	// 	{NewValueLiteral(0.5), NewValueLiteral(1.0), NewValueLiteral(1.0)},
	// 	{NewValueLiteral(1), NewValueLiteral(1.0), NewValueLiteral(-1.0)},
	// }
	// ys := []*Value{
	// 	NewValueLiteral(1),
	// 	NewValueLiteral(-1),
	// 	NewValueLiteral(-1),
	// 	NewValueLiteral(1),
	// }

	// learningRate := 0.05
	// for epoch := range 1000 {
	// 	// Forward Pass
	// 	loss := NewValueLiteral(0)
	// 	for index, i := range xs {
	// 		output := layer.Call(i)[0]
	// 		target := ys[index]

	// 		loss = loss.Add(target.Sub(output).Pow(2))
	// 	}
	// 	fmt.Printf("Epoch = %d, Loss = %.12f\n", epoch, loss.Value)

	// 	// Backward Pass
	// 	params := layer.Parameters()
	// 	for _, param := range params {
	// 		param.Grad = 0
	// 	}
	// 	loss.Backward()

	// 	// Update
	// 	for j := range params {
	// 		params[j].Value -= params[j].Grad * learningRate
	// 	}

	// }
}
