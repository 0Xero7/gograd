package nn

import (
	"gograd/ng"
)

type NeuronTensor struct {
	Dimension int
	Weights   *ng.Tensor
	Bias      *ng.Tensor
}

func NewNeuronTensor(dim int) *NeuronTensor {
	n := new(NeuronTensor)
	n.Dimension = dim
	n.Weights = ng.TensorConst(0, dim)
	n.Bias = ng.TensorConst(0, dim)
	return n
}

func (n *NeuronTensor) Call(inputs *ng.Tensor) *ng.Tensor {
	if inputs.Len() != n.Dimension {
		panic("Input dimensions don't match neuron dimensions")
	}

	// act := n.Bias
	// for i := range inputs {
	// 	act = act.Add(n.Weights[i].Mul(inputs[i]))
	// }

	return inputs.Mul(n.Weights).Add(n.Bias)
}

func (n *NeuronTensor) Parameters() []*ng.Tensor {
	return []*ng.Tensor{n.Weights, n.Bias}
}
