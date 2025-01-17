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
	n.Bias = ng.Scalar(0)
	n.Weights.RequiresOptimization = true
	n.Bias.RequiresOptimization = true
	return n
}

// Takes in a multidimensional tensor, but returns a scalar (0-Dimensional Tensor)
func (n *NeuronTensor) Call(inputs *ng.Tensor) *ng.Tensor {
	if inputs.Len() != n.Dimension {
		panic("Input dimensions don't match neuron dimensions")
	}

	return inputs.Mul(n.Weights).Sum().Add(n.Bias)
}

func (n *NeuronTensor) Parameters() []*ng.Tensor {
	return []*ng.Tensor{n.Weights, n.Bias}
}
