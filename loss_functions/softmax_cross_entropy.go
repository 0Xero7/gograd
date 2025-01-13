package lossfunctions

import "gograd/ng"

func SoftmaxCrossEntropy(logits []*ng.Value, class int) *ng.Value {
	maxLogit := logits[0].Value
	for _, logit := range logits {
		maxLogit = max(maxLogit, logit.Value)
	}

	maxLogitNode := ng.NewValueLiteral(maxLogit)
	exps := make([]*ng.Value, 0)
	sumExp := ng.NewValueLiteral(0)
	for i := range logits {
		shifted := logits[i].Sub(maxLogitNode)
		exp := shifted.Exp()
		exps = append(exps, exp)
		sumExp = sumExp.Add(exp)
	}

	probs := make([]*ng.Value, len(logits))
	for i := range exps {
		probs[i] = exps[i].Div(sumExp)
	}

	loss := probs[class].Log().Negate()
	return loss
}
