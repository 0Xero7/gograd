package tensorlayers

import (
	"gograd/ng"
	"gograd/nn"
)

type ActivationLayer struct {
	Dim  int
	Type ActivationFunction

	CachedOutputs *ng.Tensor
}

// Constructor
func ReLu(dims int) nn.TensorLayer {
	layer := new(ActivationLayer)
	// layer.CachedOutputs = make([]*ng.Value, dims)
	layer.Dim = dims
	layer.Type = relu
	return layer
}

// Constructor
func Tanh(dims int) nn.TensorLayer {
	layer := new(ActivationLayer)
	// layer.CachedOutputs = make([]*ng.Value, dims)
	layer.Dim = dims
	layer.Type = tanh
	return layer
}

// // Constructor
// func Sigmoid(dims int) nn.Layer {
// 	layer := new(ActivationLayer)
// 	layer.CachedOutputs = make([]*ng.Value, dims)
// 	layer.Dim = dims
// 	layer.Type = sigmoid
// 	return layer
// }

func (l *ActivationLayer) Call(inputs *ng.Tensor) *ng.Tensor {
	// if inputs.Dim() != l.Dim {
	// 	panic("Input dimensions don't match neuron dimensions")
	// }

	// for i := range l.Dim {
	switch l.Type {
	case relu:
		l.CachedOutputs = inputs.ReLu()
	case tanh:
		l.CachedOutputs = inputs.Tanh()
	case sigmoid:
		l.CachedOutputs = inputs.Sigmoid()
	}
	// }
	return l.CachedOutputs
}

func (l *ActivationLayer) Parameters() []*ng.Tensor {
	return []*ng.Tensor{}
}

func (l *ActivationLayer) ParameterCount() int {
	return 0
}

func (l *ActivationLayer) FanOut() int {
	return l.Dim
}
