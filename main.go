package main

func main() {
	x1 := NewValueLiteral(2.0)
	x2 := NewValueLiteral(0.0)

	w1 := NewValueLiteral(-3.0)
	w2 := NewValueLiteral(1.0)

	b := NewValueLiteral(6.88137535870195432)

	x1w1 := x1.Mul(w1)
	x2w2 := x2.Mul(w2)

	x1w1x2w1 := x1w1.Add(x2w2)
	n := x1w1x2w1.Add(b)

	// o := n.Tanh()

	e := n.Mul(NewValueLiteral(2.0)).Exp()
	o := (e.Sub(NewValueLiteral(1))).Div(e.Add(NewValueLiteral(1)))

	o.Backward()

	trace(o)
}
