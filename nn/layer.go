package nn

import "gograd/ng"

type Layer interface {
	Call(inputs []*ng.Value) []*ng.Value
	Parameters() []*ng.Value

	FanOut() int
}
