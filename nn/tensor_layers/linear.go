package tensorlayers

import (
	"gograd/ng"
	"gograd/nn"
	"gograd/nn/initializers"
	"math"
)

type LinearLayer struct {
	NeuronDim int
	OutDim    int

	Neurons []*nn.NeuronTensor
}

// Constructor
func Linear(neuronDim int, outDim int, init initializers.Initializer) nn.TensorLayer {
	layer := new(LinearLayer)
	layer.NeuronDim = neuronDim
	layer.Neurons = make([]*nn.NeuronTensor, outDim)
	layer.OutDim = outDim
	factor := 1.0 / math.Sqrt(float64(neuronDim))

	for i := range outDim {
		layer.Neurons[i] = nn.NewNeuronTensor(neuronDim)
		for j := range layer.Neurons[i].Weights.Value {
			layer.Neurons[i].Weights.Value[j] = init.Sample(neuronDim, outDim) * factor
		}
	}
	return layer
}

func (l *LinearLayer) Call(inputs *ng.Tensor) *ng.Tensor {
	if inputs.Len() != l.NeuronDim {
		panic("Input dimensions don't match neuron dimensions")
	}

	// Compute each neuron's output
	outputs := make([]float64, l.OutDim)
	results := make([]*ng.Tensor, l.OutDim)

	// Compute remaining neurons and stack their outputs
	for i := range l.OutDim {
		neuronOutput := l.Neurons[i].Call(inputs)
		outputs[i] = neuronOutput.Value[0]
		results[i] = neuronOutput
	}

	// Create output tensor with all neuron outputs
	out := ng.NewTensorFlatWith(outputs, []int{l.OutDim}, "linear_output", results...)

	// Define backward pass to distribute gradients
	out.LocalBackward = func() {
		for i := 0; i < l.OutDim; i++ {
			// Propagate gradients back to each neuron's output
			results[i].Grad[0] += out.Grad[i]
		}
	}

	return out
}

// func (l *LinearLayer) Call(inputs *ng.Tensor) *ng.Tensor {
// 	if inputs.Len() != l.NeuronDim {
// 		panic("Input dimensions don't match neuron dimensions")
// 	}

// 	// out := make([]*ng.Tensor, l.OutDim)
// 	// for i := range l.OutDim {
// 	// 	out[i] = l.Neurons[i].Call(inputs)
// 	// }
// 	return
// }

func (l *LinearLayer) Parameters() []*ng.Tensor {
	params := make([]*ng.Tensor, 0)
	for _, n := range l.Neurons {
		params = append(params, n.Parameters()...)
	}
	return params
}

func (l *LinearLayer) FanOut() int {
	return l.OutDim
}
