package main

import (
	"math"
	"math/rand"
)

type Activation int

const (
	Linear Activation = iota
	Tanh
	ReLu
)

type Neuron struct {
	Dimension  int
	Activation Activation
	Weights    []*Value
	Bias       *Value
}

func NewNeuron(dim int, activation Activation) *Neuron {
	n := new(Neuron)
	n.Dimension = dim
	n.Activation = activation
	n.Weights = make([]*Value, dim)
	for i := range n.Weights {
		if activation != ReLu {
			n.Weights[i] = NewValueLiteral((rand.Float64() - 0.5) * 2)
		} else {
			scale := math.Sqrt(2.0 / float64(dim))
			n.Weights[i] = NewValueLiteral((rand.Float64()*2 - 1) * scale)
		}
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

	switch n.Activation {
	case Linear:
		return act
	case ReLu:
		return act.ReLu()
	case Tanh:
		return act.Tanh()

	default:
		panic("Unsupported activatin function")
	}
}

func (n *Neuron) Parameters() []*Value {
	params := make([]*Value, 0)
	params = append(params, n.Weights...)
	params = append(params, n.Bias)
	return params
}
