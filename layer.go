package main

type Layer struct {
	NeuronDim int
	OutDim    int

	Neurons []*Neuron
}

func NewLayer(neuronDim int, outDim int) *Layer {
	layer := new(Layer)
	layer.NeuronDim = neuronDim
	layer.Neurons = make([]*Neuron, outDim)
	layer.OutDim = outDim

	for i := range outDim {
		layer.Neurons[i] = NewNeuron(neuronDim)
	}
	return layer
}

func (l *Layer) Call(inputs []*Value) []*Value {
	if len(inputs) != l.NeuronDim {
		panic("Input dimensions don't match neuron dimensions")
	}

	out := make([]*Value, l.OutDim)
	for i := range l.OutDim {
		out[i] = l.Neurons[i].Call(inputs)
	}
	return out
}

func (l *Layer) Parameters() []*Value {
	params := make([]*Value, 0)
	for _, n := range l.Neurons {
		params = append(params, n.Parameters()...)
	}
	return params
}
