package nn

import (
	"gograd/ng"
)

type MLP struct {
	InputDim         int
	OutputDim        int
	Dims             []int
	TotalLayersCount int

	Layers []Layer
}

func NewMLP(inputDims int, layers []Layer) *MLP {
	mlp := new(MLP)
	mlp.InputDim = inputDims
	mlp.OutputDim = layers[len(layers)-1].FanOut()
	mlp.Dims = make([]int, 0)
	mlp.Dims = append(mlp.Dims, inputDims)
	for _, layer := range layers {
		mlp.Dims = append(mlp.Dims, layer.FanOut())
	}
	mlp.TotalLayersCount = len(layers) + 1
	mlp.Layers = layers
	return mlp
}

func (m *MLP) Call(inputs []*ng.Value) []*ng.Value {
	t := inputs
	for _, layer := range m.Layers {
		t = layer.Call(t)
	}
	return t
}

func (m *MLP) Parameters() []*ng.Value {
	params := make([]*ng.Value, 0)
	for _, n := range m.Layers {
		params = append(params, n.Parameters()...)
	}
	return params
}

// Get predictions for a single input
func (mlp *MLP) Predict(input []*ng.Value) int {
	// Forward pass
	values := mlp.Call(input)

	// Find max logit (don't need full softmax for prediction)
	maxIdx := 0
	maxVal := values[0].Value
	for i, v := range values {
		if v.Value > maxVal {
			maxVal = v.Value
			maxIdx = i
		}
	}

	ng.TValuePool.Reset()

	return maxIdx
}
