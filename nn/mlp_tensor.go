package nn

import (
	"gograd/ng"
)

type MLPTensor struct {
	InputDim         int
	OutputDim        int
	Dims             []int
	TotalLayersCount int

	Layers []TensorLayer
}

func NewMLPTensor(inputDims int, layers []TensorLayer) *MLPTensor {
	mlp := new(MLPTensor)
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

func (m *MLPTensor) Call(inputs *ng.Tensor) *ng.Tensor {
	t := inputs
	for _, layer := range m.Layers {
		// fmt.Printf(">>>>> Layer %d\ninput shape: %v", idx, layer.Parameters()[0].Shape)
		t = layer.Call(t)
		// fmt.Printf(" output shape: %v\n\n", t.Shape)
	}
	return t
}

func (m *MLPTensor) Parameters() []*ng.Tensor {
	params := make([]*ng.Tensor, 0)
	for _, n := range m.Layers {
		params = append(params, n.Parameters()...)
	}
	return params
}

func (m *MLPTensor) ParameterCount() int {
	count := 0
	for _, n := range m.Layers {
		count += n.ParameterCount()
	}
	return count
}

// Get predictions for a single input
func (mlp *MLPTensor) Predict(input *ng.Tensor) int {
	if input.Len() == 0 {
		panic("No input provided for prediction")
	}

	// Forward pass with first input
	output := mlp.Call(input)

	highestIndex := -1
	highVal := -10000000000000.0

	for i := range output.Len() {
		val := output.Value[i]
		if val > highVal {
			highVal = val
			highestIndex = i
		}
	}

	return highestIndex

	// // Output should be a tensor of shape [OutputDim]
	// if len(output.Shape) != 1 || output.Shape[0] != mlp.OutputDim {
	// 	panic("Invalid output shape from network")
	// }

	// // Find max logit
	// maxIdx := 0
	// maxVal := output.Value[0]
	// for i := 1; i < mlp.OutputDim; i++ {
	// 	if output.Value[i] > maxVal {
	// 		maxVal = output.Value[i]
	// 		maxIdx = i
	// 	}
	// }

	// ng.TTensorPool.Reset()
	// return maxIdx
}
