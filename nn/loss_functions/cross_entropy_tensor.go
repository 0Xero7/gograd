package lossfunctions

import (
	"gograd/ng"
)

func CrossEntropyTensor(probabilities *ng.Tensor, class int) *ng.Tensor {
	return probabilities.Log().Negate().Choose(class)
}

// func BatchCrossEntropy(probabilities [][]*ng.Value, classes []int) *ng.Value {
// 	if len(probabilities) != len(classes) {
// 		panic("BatchCrossEntropy inputs have different length")
// 	}

// 	n := ng.NewValueLiteral(float64(len(probabilities)))
// 	sum := CrossEntropy(probabilities[0], classes[0])
// 	for i := 1; i < len(probabilities); i++ {
// 		sum = sum.Add(CrossEntropy(probabilities[i], classes[i]))
// 	}
// 	return sum.Div(n)
// }
