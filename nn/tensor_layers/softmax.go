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
	return inputs.SoftMax(1)
}

func (l *SoftmaxLayer) Parameters() []*ng.Tensor {
	return []*ng.Tensor{}
}

func (l *SoftmaxLayer) ParameterCount() int {
	return 0
}

func (l *SoftmaxLayer) FanOut() int {
	return l.Dim
}
