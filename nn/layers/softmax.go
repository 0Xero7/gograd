package layers

import "gograd/ng"

type SoftMaxLayer struct {
	Dim           int
	CachedOutputs []*ng.Value
}

func SoftMax(dim int) *SoftMaxLayer {
	return &SoftMaxLayer{
		Dim:           dim,
		CachedOutputs: make([]*ng.Value, dim),
	}
}

func (s *SoftMaxLayer) Call(inputs []*ng.Value) []*ng.Value {
	if len(inputs) != s.Dim {
		panic("Input dimensions don't match neuron dimensions")
	}

	maxLogit := inputs[0]
	for _, logit := range inputs[1:] {
		maxLogit = maxLogit.Max(logit)
	}

	var sumExp *ng.Value = nil
	for i, logit := range inputs {
		s.CachedOutputs[i] = logit.Sub(maxLogit).Exp()

		if sumExp == nil {
			sumExp = s.CachedOutputs[i]
		} else {
			sumExp = sumExp.Add(s.CachedOutputs[i])
		}
	}

	for i := range s.CachedOutputs {
		s.CachedOutputs[i] = s.CachedOutputs[i].Div(sumExp)
	}

	return s.CachedOutputs
}

func (s *SoftMaxLayer) Parameters() []*ng.Value {
	return s.CachedOutputs
}

func (s *SoftMaxLayer) FanOut() int {
	return s.Dim
}
