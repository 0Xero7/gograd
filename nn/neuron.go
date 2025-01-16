package nn

import (
	"gograd/ng"
)

type Activation int

const (
	Linear Activation = iota
	Tanh
	ReLu
	Sigmoid
)

type Neuron struct {
	Dimension  int
	Activation Activation
	Weights    []*ng.Value
	Bias       *ng.Value
}

func NewNeuron(dim int) *Neuron {
	n := new(Neuron)
	n.Dimension = dim
	n.Weights = make([]*ng.Value, dim)
	n.Bias = ng.NewValueLiteral(0)
	return n
}

func (n *Neuron) Call(inputs []*ng.Value) *ng.Value {
	if len(inputs) != n.Dimension {
		panic("Input dimensions don't match neuron dimensions")
	}

	act := n.Bias
	for i := range inputs {
		act = act.Add(n.Weights[i].Mul(inputs[i]))
	}

	return act
}

func (n *Neuron) Parameters() []*ng.Value {
	params := make([]*ng.Value, 0)
	params = append(params, n.Weights...)
	params = append(params, n.Bias)
	return params
}
