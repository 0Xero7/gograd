package lossfunctions

import (
	"gograd/ng"
)

func SoftmaxCrossEntropy(logits []*ng.Value, class int) *ng.Value {
	maxLogit := logits[0]
	for _, logit := range logits[1:] {
		maxLogit = maxLogit.Max(logit)
	}

	// maxLogitNode := ng.NewValueLiteral(maxLogit)
	exps := make([]*ng.Value, 0)
	sumExp := ng.NewValueLiteral(0)
	for i := range logits {
		shifted := logits[i].Sub(maxLogit)
		exp := shifted.Exp()
		exps = append(exps, exp)
		sumExp = sumExp.Add(exp)
	}

	probs := make([]*ng.Value, len(logits))
	for i := range exps {
		probs[i] = exps[i].Div(sumExp)
	}

	loss := probs[class].Log().Negate()

	// fmt.Printf("Before softmax logits: %v\n", logits) // Raw outputs before softmax
	// fmt.Printf("Probabilities: %v\n", probs)          // After softmax
	// fmt.Printf("Loss value: %f\n", loss.Value)        // Final loss

	return loss
}
