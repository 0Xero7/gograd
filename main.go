package main

import "fmt"

func main() {
	layer := NewMLP(3, []int{4, 4}, 1)

	xs := [][]*Value{
		{NewValueLiteral(2.0), NewValueLiteral(3.0), NewValueLiteral(-1.0)},
		{NewValueLiteral(3.0), NewValueLiteral(-1.0), NewValueLiteral(0.5)},
		{NewValueLiteral(0.5), NewValueLiteral(1.0), NewValueLiteral(1.0)},
		{NewValueLiteral(1), NewValueLiteral(1.0), NewValueLiteral(-1.0)},
	}
	ys := []*Value{
		NewValueLiteral(1),
		NewValueLiteral(-1),
		NewValueLiteral(-1),
		NewValueLiteral(1),
	}

	learningRate := 0.05
	for epoch := range 1000 {
		// Forward Pass
		loss := NewValueLiteral(0)
		for index, i := range xs {
			output := layer.Call(i)[0]
			target := ys[index]

			loss = loss.Add(target.Sub(output).Pow(2))
		}
		fmt.Printf("Epoch = %d, Loss = %.12f\n", epoch, loss.Value)

		// Backward Pass
		params := layer.Parameters()
		for _, param := range params {
			param.Grad = 0
		}
		loss.Backward()

		// Update
		for j := range params {
			params[j].Value -= params[j].Grad * learningRate
		}
	}
}
