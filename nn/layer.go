package nn

import "gograd/ng"

type Layer struct {
	NeuronDim int
	OutDim    int

	Neurons []*Neuron
}

func NewLayer(neuronDim int, outDim int, activation Activation) *Layer {
	layer := new(Layer)
	layer.NeuronDim = neuronDim
	layer.Neurons = make([]*Neuron, outDim)
	layer.OutDim = outDim

	for i := range outDim {
		layer.Neurons[i] = NewNeuron(neuronDim, activation)
	}
	return layer
}

func (l *Layer) Call(inputs []*ng.Value) []*ng.Value {
	if len(inputs) != l.NeuronDim {
		panic("Input dimensions don't match neuron dimensions")
	}

	out := make([]*ng.Value, l.OutDim)
	for i := range l.OutDim {
		out[i] = l.Neurons[i].Call(inputs)
	}
	return out
}

func (l *Layer) Parameters() []*ng.Value {
	params := make([]*ng.Value, 0)
	for _, n := range l.Neurons {
		params = append(params, n.Parameters()...)
	}
	return params
}
