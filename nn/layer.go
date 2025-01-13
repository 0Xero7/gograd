package nn

import "gograd/ng"

// type LayerX struct {
// 	NeuronDim int
// 	OutDim    int

// 	Neurons []*Neuron
// }

type Layer interface {
	Call(inputs []*ng.Value) []*ng.Value
	Parameters() []*ng.Value

	FanOut() int
}

// func NewLayer(neuronDim int, outDim int) *LayerX {
// 	layer := new(LayerX)
// 	layer.NeuronDim = neuronDim
// 	layer.Neurons = make([]*Neuron, outDim)
// 	layer.OutDim = outDim

// 	for i := range outDim {
// 		layer.Neurons[i] = NewNeuron(neuronDim)
// 	}
// 	return layer
// }

// func (l *LayerX) Call(inputs []*ng.Value) []*ng.Value {
// 	if len(inputs) != l.NeuronDim {
// 		panic("Input dimensions don't match neuron dimensions")
// 	}

// 	out := make([]*ng.Value, l.OutDim)
// 	for i := range l.OutDim {
// 		out[i] = l.Neurons[i].Call(inputs)
// 	}
// 	return out
// }

// func (l *LayerX) Parameters() []*ng.Value {
// 	params := make([]*ng.Value, 0)
// 	for _, n := range l.Neurons {
// 		params = append(params, n.Parameters()...)
// 	}
// 	return params
// }
