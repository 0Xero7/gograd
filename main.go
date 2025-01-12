package main

func main() {
	x1 := NewValueLiteral(2.0)
	x1.Label = "x1"
	x2 := NewValueLiteral(0.0)
	x2.Label = "x2"

	w1 := NewValueLiteral(-3.0)
	w1.Label = "w1"
	w2 := NewValueLiteral(1.0)
	w2.Label = "w2"

	b := NewValueLiteral(6.88137535870195432)
	b.Label = "b"

	x1w1 := x1.Mul(w1)
	x1w1.Label = "x1w1"
	x2w2 := x2.Mul(w2)
	x2w2.Label = "x2w2"

	x1w1x2w1 := x1w1.Add(x2w2)
	x1w1x2w1.Label = "x1w1x2w1"
	n := x1w1x2w1.Add(b)
	n.Label = "n"

	// n.Backward()
	// trace(n)

	e := n.Mul(NewValueLiteral(2.0)).Exp()
	e.Label = "e"
	o := (e.Sub(NewValueLiteral(1))).Div(e.Add(NewValueLiteral(1)))
	o.Label = "o"
	o.Backward()
	trace(o)
}
