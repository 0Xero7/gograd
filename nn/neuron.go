package nn

import (
	"gograd/ng"
	"math/rand"
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
	for i := range n.Weights {
		// if activation != ReLu {
		n.Weights[i] = ng.NewValueLiteral((rand.Float64() - 0.5) * 2)
		// } else {
		// 	scale := math.Sqrt(2.0 / float64(dim))
		// 	n.Weights[i] = ng.NewValueLiteral((rand.Float64()*2 - 1) * scale)
		// }
	}
	n.Bias = ng.NewValueLiteral((rand.Float64() - 0.5) * 2)
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
	// switch n.Activation {
	// case Linear:
	// case ReLu:
	// 	return act.ReLu()
	// case Tanh:
	// 	return act.Tanh()
	// case Sigmoid:
	// 	return act.Sigmoid()

	// default:
	// 	panic("Unsupported activatin function")
	// }
}

func (n *Neuron) Parameters() []*ng.Value {
	params := make([]*ng.Value, 0)
	params = append(params, n.Weights...)
	params = append(params, n.Bias)
	return params
}
