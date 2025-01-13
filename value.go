package main

import (
	"fmt"
	"math"
	"math/rand"
)

type Value struct {
	Id            int
	Value         float64
	Grad          float64
	LocalBackward func()
	Op            string
	Children      []*Value
	Label         string
}

func NewValueLiteral(value float64) *Value {
	v := new(Value)
	v.Id = rand.Int()
	v.Label = fmt.Sprint(v.Id)

	v.Value = value
	v.Grad = 0
	v.Op = "X"
	v.Children = make([]*Value, 0)
	return v
}

func NewValue(value float64, op string, children []*Value) *Value {
	v := new(Value)
	v.Id = rand.Int()
	v.Label = fmt.Sprint(v.Id)

	v.Value = value
	v.Grad = 0
	v.Op = op
	v.Children = make([]*Value, 0, len(children))

	seen := make(map[*Value]bool)
	for _, val := range children {
		if !seen[val] {
			v.Children = append(v.Children, val)
			seen[val] = true
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
		v.Grad += -1.0 * t.Grad
	}
	return t
}

func (v *Value) Add(other *Value) *Value {
	t := NewValue(v.Value+other.Value, "+", []*Value{v, other})
	t.LocalBackward = func() {
		v.Grad += t.Grad
		other.Grad += t.Grad
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
	}
	return t
}

func (v *Value) Div(other *Value) *Value {
	return v.Mul(other.Pow(-1))
}

func (v *Value) Log() *Value {
	epsilon := 1e-7
	safeValue := math.Max(v.Value, epsilon)
	t := NewValue(math.Log(safeValue), "log", []*Value{v})
	t.LocalBackward = func() {
		v.Grad += (1.0 / safeValue) * t.Grad
	}
	return t
}

func (v *Value) Tanh() *Value {
	t := NewValue(math.Tanh(v.Value), "tanh", []*Value{v})
	t.LocalBackward = func() {
		v.Grad += (1.0 - math.Pow(t.Value, 2.0)) * t.Grad
	}
	return t
}

func (v *Value) ReLu() *Value {
	t := NewValue(max(0, v.Value), "relu", []*Value{v})
	t.LocalBackward = func() {
		grad := 0.0
		if t.Value > 0 {
			grad = 1.0
		}
		v.Grad += grad * t.Grad
	}
	return t
}

func (v *Value) Sigmoid() *Value {
	val := 1.0 / (1.0 + math.Exp(-v.Value))
	t := NewValue(val, "sigmoid", []*Value{v})
	t.LocalBackward = func() {
		v.Grad += (t.Value) * (1.0 - v.Value) * t.Grad
	}
	return t
}

func (v *Value) Exp() *Value {
	t := NewValue(math.Exp(v.Value), "exp", []*Value{v})
	t.LocalBackward = func() {
		v.Grad += t.Value * t.Grad
	}
	return t
}

func (v *Value) Pow(other float64) *Value {
	t := NewValue(math.Pow(v.Value, other), "pow", []*Value{v})
	t.LocalBackward = func() {
		v.Grad += other * math.Pow(v.Value, other-1) * t.Grad
	}
	return t
}

func (v *Value) PerformBackward() {
	if v.LocalBackward != nil {
		v.LocalBackward()
	}
}

func dfs(root *Value, visited *map[int]bool, collect *[]*Value) {
	if _, found := (*visited)[root.Id]; found {
		return
	}

	(*visited)[root.Id] = true
	for _, child := range root.Children {
		dfs(child, visited, collect)
	}
	*collect = append(*collect, root)
}

func (v *Value) Backward() {
	v.Grad = 1.0
	visited := make(map[int]bool)
	collect := make([]*Value, 0)

	dfs(v, &visited, &collect)
	for i := len(collect) - 1; i >= 0; i-- {
		collect[i].PerformBackward()
	}

	collect = nil
	visited = nil
}

func (v *Value) Clear() {
	v.Children = nil
	v.LocalBackward = nil
	v.Grad = 0
}
