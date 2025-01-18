package tensorlayers

import (
	"gograd/ng"
	"gograd/nn"
)

type SoftmaxLayer struct {
	Dim int
}

func SoftMax(dim int) nn.TensorLayer {
	layer := new(SoftmaxLayer)
	layer.Dim = dim
	return layer
}

func (l *SoftmaxLayer) Call(inputs *ng.Tensor) *ng.Tensor {
	return inputs.SoftMax(-1)
	// // Find max for numerical stability
	// maxVal := inputs.Value[0]
	// for i := 1; i < len(inputs.Value); i++ {
	// 	if inputs.Value[i] > maxVal {
	// 		maxVal = inputs.Value[i]
	// 	}
	// }

	// // Compute exp(x - max) for each input
	// expSum := 0.0
	// exps := make([]float64, len(inputs.Value))
	// for i := range inputs.Value {
	// 	exps[i] = math.Exp(inputs.Value[i] - maxVal)
	// 	expSum += exps[i]
	// }

	// // Normalize to get probabilities
	// outputs := make([]float64, len(inputs.Value))
	// for i := range outputs {
	// 	outputs[i] = exps[i] / expSum
	// }

	// out := ng.NewTensorFlatWith(outputs, []int{len(outputs)}, "softmax_output", inputs)

	// // Define backward pass for softmax
	// out.LocalBackward = func() {
	// 	for i := range outputs {
	// 		for j := range outputs {
	// 			if i == j {
	// 				inputs.Grad[i] += out.Grad[i] * outputs[i] * (1 - outputs[i])
	// 			} else {
	// 				inputs.Grad[i] += -out.Grad[j] * outputs[i] * outputs[j]
	// 			}
	// 		}
	// 	}
	// }

	// return out
}

func (l *SoftmaxLayer) Parameters() []*ng.Tensor {
	return []*ng.Tensor{}
}

func (l *SoftmaxLayer) FanOut() int {
	return l.Dim
}
