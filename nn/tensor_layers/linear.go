package tensorlayers

import (
	"fmt"
	"gograd/ng"
	"gograd/nn/initializers"
	"gograd/utils"
	"math"
)

type LinearLayer struct {
	DimIn  int
	DimOut int

	Weights *ng.Tensor
	Bias    *ng.Tensor
}

// Constructor
func Linear(inDim int, outDim int, init initializers.Initializer) *LinearLayer {
	layer := new(LinearLayer)
	layer.DimIn = inDim
	layer.DimOut = outDim

	layer.Weights = ng.TensorOnes(inDim, outDim)
	layer.Weights.RequiresOptimization = true
	factor := 1.0 / math.Sqrt(float64(inDim))
	for x := range inDim {
		for y := range outDim {
			layer.Weights.Set(init.Sample(inDim, outDim)*factor, x, y)
		}
	}
	layer.Bias = ng.TensorConst(0, outDim)
	layer.Bias.RequiresOptimization = true

	return layer
}

func (l *LinearLayer) Call(inputs *ng.Tensor) *ng.Tensor {
	utils.AssertTrue(inputs.Size%l.DimIn == 0, fmt.Sprint("Input tensor of shape ", inputs.Shape, " is not compatible with layer of shape ", l.Weights.Shape))

	inputFirstDim := inputs.Size / l.DimIn
	return inputs.ReshapeOut(inputFirstDim, l.DimIn).MatMul(l.Weights).Add(l.Bias)
}

func (l *LinearLayer) Parameters() []*ng.Tensor {
	return []*ng.Tensor{l.Weights, l.Bias}
}

func (l *LinearLayer) ParameterCount() int {
	return l.Weights.Len() + l.Bias.Len()
}

func (l *LinearLayer) FanOut() int {
	return l.DimOut
}
