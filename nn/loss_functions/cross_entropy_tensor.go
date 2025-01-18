package lossfunctions

import (
	"gograd/ng"
	"gograd/utils"
	"math"
	"slices"
)

func CrossEntropyTensor(probabilities *ng.Tensor, class int) *ng.Tensor {
	// fmt.Printf(">> Probs: %v, Class: %d\n", probabilities.Value, class)
	// fmt.Printf(">> Log probs: %v\n", probabilities.Log().Value)
	return probabilities.Log().Negate().Choose(class)
}

func BatchCrossEntropyTensor(probabilities []*ng.Tensor, classes []int) *ng.Tensor {
	if len(probabilities) != len(classes) {
		panic("BatchCrossEntropy inputs have different length")
	}

	n := ng.Scalar(float64(len(probabilities)))
	sum := CrossEntropyTensor(probabilities[0], classes[0])
	for i := 1; i < len(probabilities); i++ {
		sum = sum.Add(CrossEntropyTensor(probabilities[i], classes[i]))
	}
	return sum.Div(n)
}

// --------------------------------------------------------------------------------------------------

// Takes in logits, target output tensors and outputs a 0-D (scalar) tensor with the loss
// Assumes first dimension is the batch dimension. *Mean* cross entropy loss is computed.
func CrossEntropyLoss(input *ng.Tensor, target *ng.Tensor) *ng.Tensor {
	utils.AssertTrue(slices.Equal(input.Shape, target.Shape), "CrossEntropyLoss input and target tensors have different shapes")

	// fmt.Println("input=", input)
	epsilon := 1e-7

	lastIndex := input.Dim() - 1
	sliceIndices := make([]int, input.Dim())
	loss := 0.0
	for {
		for i := range input.Shape[lastIndex] {
			sliceIndices[lastIndex] = i
			// fmt.Println(">>", input.Get(sliceIndices...), -math.Log(input.Get(sliceIndices...)+epsilon))
			loss += -math.Log(input.Get(sliceIndices...)+epsilon) * target.Get(sliceIndices...)
		}

		if !ng.NextSlicedIndex(input, lastIndex, &sliceIndices) {
			break
		}
	}
	N := (float64(input.Len()) / float64(input.Shape[lastIndex]))
	loss /= N

	out := ng.NewTensorFlatWith([]float64{loss}, []int{1}, "crossentropyloss", input)

	out.LocalBackward = func() {
		for position := range input.Grad {
			// fmt.Println(">>", position)
			// input.Grad[position] = 6
			input.Grad[position] += -(target.Value[position] / (input.Value[position] + epsilon)) * out.Grad[0] / N
			// input.Grad[position] = -(target.Value[position] / (input.Value[position] + epsilon)) * out.Grad[0]
			// input.Grad[position] += ((input.Value[position] - target.Value[position]) * out.Grad[0]) / N
		}
		// fmt.Println(input)
	}

	return out
}
