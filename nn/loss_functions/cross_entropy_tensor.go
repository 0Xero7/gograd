package lossfunctions

import (
	"gograd/ng"
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
