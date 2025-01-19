package nn

import "gograd/ng"

type TensorLayer interface {
	Call(inputs *ng.Tensor) *ng.Tensor
	Parameters() []*ng.Tensor
	ParameterCount() int

	FanOut() int
}
