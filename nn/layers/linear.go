package layers

import (
	"gograd/ng"
	"gograd/nn"
)

type LinearLayer struct {
	NeuronDim int
	OutDim    int

	Neurons []*nn.Neuron
}

// Constructor
func Linear(neuronDim int, outDim int) nn.Layer {
	layer := new(LinearLayer)
	layer.NeuronDim = neuronDim
	layer.Neurons = make([]*nn.Neuron, outDim)
	layer.OutDim = outDim

	for i := range outDim {
		layer.Neurons[i] = nn.NewNeuron(neuronDim)
	}
	return layer
}

func (l *LinearLayer) Call(inputs []*ng.Value) []*ng.Value {
	if len(inputs) != l.NeuronDim {
		panic("Input dimensions don't match neuron dimensions")
	}

	out := make([]*ng.Value, l.OutDim)
	for i := range l.OutDim {
		out[i] = l.Neurons[i].Call(inputs)
	}
	return out
}

func (l *LinearLayer) Parameters() []*ng.Value {
	params := make([]*ng.Value, 0)
	for _, n := range l.Neurons {
		params = append(params, n.Parameters()...)
	}
	return params
}

func (l *LinearLayer) FanOut() int {
	return l.OutDim
}
