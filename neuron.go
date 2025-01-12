package main

import "math/rand"

type Neuron struct {
	Dimension int
	Weights   []*Value
	Bias      *Value
}

func NewNeuron(dim int) *Neuron {
	n := new(Neuron)
	n.Dimension = dim
	n.Weights = make([]*Value, dim)
	for i := range n.Weights {
		n.Weights[i] = NewValueLiteral((rand.Float64() - 0.5) * 2)
	}
	n.Bias = NewValueLiteral((rand.Float64() - 0.5) * 2)
	return n
}

func (n *Neuron) Call(inputs []*Value) *Value {
	if len(inputs) != n.Dimension {
		panic("Input dimensions don't match neuron dimensions")
	}

	act := n.Bias
	for i := range inputs {
		act = act.Add(n.Weights[i].Mul(inputs[i]))
	}

	return act.Tanh()
}

func (n *Neuron) Parameters() []*Value {
	params := make([]*Value, 0)
	params = append(params, n.Weights...)
	params = append(params, n.Bias)
	return params
}
