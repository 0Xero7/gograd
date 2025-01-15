package layers

import (
	"gograd/ng"
	"gograd/nn"
)

type ActivationLayer struct {
	Dim  int
	Type ActivationFunction

	CachedOutputs []*ng.Value
}

// Constructor
func ReLu(dims int) nn.Layer {
	layer := new(ActivationLayer)
	layer.CachedOutputs = make([]*ng.Value, dims)
	layer.Dim = dims
	layer.Type = relu
	return layer
}

// Constructor
func Tanh(dims int) nn.Layer {
	layer := new(ActivationLayer)
	layer.CachedOutputs = make([]*ng.Value, dims)
	layer.Dim = dims
	layer.Type = tanh
	return layer
}

// Constructor
func Sigmoid(dims int) nn.Layer {
	layer := new(ActivationLayer)
	layer.CachedOutputs = make([]*ng.Value, dims)
	layer.Dim = dims
	layer.Type = sigmoid
	return layer
}

func (l *ActivationLayer) Call(inputs []*ng.Value) []*ng.Value {
	if len(inputs) != l.Dim {
		panic("Input dimensions don't match neuron dimensions")
	}

	for i := range l.Dim {
		switch l.Type {
		case relu:
			l.CachedOutputs[i] = inputs[i].ReLu()
		case tanh:
			l.CachedOutputs[i] = inputs[i].Tanh()
		case sigmoid:
			l.CachedOutputs[i] = inputs[i].Sigmoid()
		}
	}
	return l.CachedOutputs
}

func (l *ActivationLayer) Parameters() []*ng.Value {
	return l.CachedOutputs
}

func (l *ActivationLayer) FanOut() int {
	return l.Dim
}
