package tensorlayers

import "gograd/ng"

type SoftMaxLayer struct {
	Dim           int
	CachedOutputs *ng.Tensor
}

func SoftMax(dim int) *SoftMaxLayer {
	return &SoftMaxLayer{
		Dim: dim,
	}
}

func (s *SoftMaxLayer) Call(inputs *ng.Tensor) *ng.Tensor {
	if inputs.Len() != s.Dim {
		panic("Input dimensions don't match neuron dimensions")
	}

	biggest := inputs.MaximumElement()
	biggest.Extend(s.Dim)

	exponents := inputs.Sub(biggest).Exp()
	expSum := exponents.Sum()
	expSum.Extend(s.Dim)
	logits := exponents.Div(expSum)

	return logits

	// Compute each neuron's output
	// outputs := make([]float64, s.Dim)
	// results := make([]*ng.Tensor, s.Dim)

	// Compute remaining neurons and stack their outputs

	// for i := range s.Dim {
	// 	neuronOutput := s.Neurons[i].Call(inputs)
	// 	outputs[i] = neuronOutput.Value[0]
	// 	results[i] = neuronOutput
	// }

	// Create output tensor with all neuron outputs
	// out := ng.NewTensorFlatWith(outputs, []int{s.Dim}, "linear_output", results...)

	// maxLogit := inputs[0]
	// for _, logit := range inputs[1:] {
	// 	maxLogit = maxLogit.Max(logit)
	// }

	// var sumExp *ng.Tensor = nil
	// // for i, logit := range inputs {
	// s.CachedOutputs = logit.Sub(maxLogit).Exp()

	// if sumExp == nil {
	// 	sumExp = s.CachedOutputs[i]
	// } else {
	// 	sumExp = sumExp.Add(s.CachedOutputs[i])
	// }
	// // }

	// for i := range s.CachedOutputs {
	// 	s.CachedOutputs[i] = s.CachedOutputs[i].Div(sumExp)
	// }

	// return s.CachedOutputs
}

func (s *SoftMaxLayer) Parameters() []*ng.Tensor {
	return []*ng.Tensor{}
}

func (s *SoftMaxLayer) FanOut() int {
	return s.Dim
}
