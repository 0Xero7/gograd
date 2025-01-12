package main

func SoftmaxCrossEntropy(logits []*Value, class int) *Value {
	maxLogit := logits[0].Value
	for _, logit := range logits {
		maxLogit = max(maxLogit, logit.Value)
	}

	maxLogitNode := NewValueLiteral(maxLogit)
	exps := make([]*Value, 0)
	sumExp := NewValueLiteral(0)
	for i := range logits {
		shifted := logits[i].Sub(maxLogitNode)
		exp := shifted.Exp()
		exps = append(exps, exp)
		sumExp = sumExp.Add(exp)
	}

	probs := make([]*Value, len(logits))
	for i := range exps {
		probs[i] = exps[i].Div(sumExp)
	}

	loss := probs[class].Log().Negate()
	return loss
}
