package main

import (
	"fmt"
	"math"
	"slices"
)

type Value struct {
	Value         float64
	Grad          float64
	LocalBackward func()
	Op            string
	Children      []*Value
}

func NewValueLiteral(value float64) *Value {
	v := new(Value)
	v.Value = value
	v.Grad = 0
	v.Op = "X"
	v.Children = make([]*Value, 0)
	return v
}

func NewValue(value float64, op string, children []*Value) *Value {
	v := new(Value)
	v.Value = value
	v.Grad = 0
	v.Op = op
	v.Children = make([]*Value, 0)

	for _, val := range children {
		if !slices.Contains(v.Children, val) {
			v.Children = append(v.Children, val)
		}
	}

	return v
}

func (v *Value) String() string {
	var opStr string
	if v.Op == "X" {
		opStr = ""
	} else {
		opStr = string(v.Op)
	}
	return fmt.Sprintf("Value(data=%.4f, grad=%.4f%s)", v.Value, v.Grad, opStr)
}

func (v *Value) Negate() *Value {
	t := NewValue(-v.Value, "neg", []*Value{v})
	t.LocalBackward = func() {
		v.Grad = -1.0 * t.Grad
		v.performBackward()
	}
	return t
}

func (v *Value) Add(other *Value) *Value {
	t := NewValue(v.Value+other.Value, "+", []*Value{v, other})
	t.LocalBackward = func() {
		v.Grad += t.Grad
		other.Grad += t.Grad
		v.performBackward()
		other.performBackward()
	}
	return t
}

func (v *Value) Sub(other *Value) *Value {
	return v.Add(other.Negate())
}

func (v *Value) Mul(other *Value) *Value {
	t := NewValue(v.Value*other.Value, "*", []*Value{v, other})
	t.LocalBackward = func() {
		v.Grad += t.Grad * other.Value
		other.Grad += t.Grad * v.Value
		v.performBackward()
		other.performBackward()
	}
	return t
}

func (v *Value) Div(other *Value) *Value {
	return v.Mul(other.Pow(-1))
}

func (v *Value) Tanh() *Value {
	t := NewValue(math.Tanh(v.Value), "tanh", []*Value{v})
	t.LocalBackward = func() {
		v.Grad = (1.0 - math.Pow(t.Value, 2.0)) * t.Grad
		v.performBackward()
	}
	return t
}

func (v *Value) Exp() *Value {
	t := NewValue(math.Exp(v.Value), "exp", []*Value{v})
	t.LocalBackward = func() {
		v.Grad += t.Value * t.Grad
		v.performBackward()
	}
	return t
}

func (v *Value) Pow(other float64) *Value {
	t := NewValue(math.Pow(v.Value, other), "pow", []*Value{v})
	t.LocalBackward = func() {
		v.Grad += other * math.Pow(v.Value, other-1) * t.Grad
		v.performBackward()
	}
	return t
}

func (v *Value) performBackward() {
	if v.LocalBackward != nil {
		v.LocalBackward()
	}
}

func (v *Value) Backward() {
	v.Grad = 1.0
	v.performBackward()
}
