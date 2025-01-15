package ng

import (
	"fmt"
	"math"
	"sync/atomic"
)

var IdGen atomic.Int64

type Value struct {
	Id            int
	Value         float64
	Grad          float64
	LocalBackward func()
	Op            string
	Children      Children
	// Label         string

	// Parents map[int]*Value
}

var TValuePool ValuePool[Value] = NewValuePool(func(index int) *Value {
	r := new(Value)
	r.Children.Clear()
	return r
})

func NewValueLiteral(value float64) *Value {
	v := TValuePool.Get() //new(Value)
	v.Id = int(IdGen.Load())
	IdGen.Add(1)
	// v.Label = fmt.Sprint(v.Id)

	v.Value = value
	v.Grad = 0
	v.Op = "X"
	v.Children.Clear()
	// if v.Children == nil {
	// 	v.Children = make([]*Value, 0)
	// } else {
	// 	v.Children = v.Children[0:0]
	// }
	// v.Parents = make(map[int]*Value)
	return v
}

func NewValue(value float64, op string) *Value {
	v := TValuePool.Get() //new(Value)
	v.Id = int(IdGen.Load())
	IdGen.Add(1)
	// v.Label = fmt.Sprint(v.Id)

	v.Value = value
	v.Grad = 0
	v.Op = op
	v.Children.Clear()
	// if v.Children == nil {
	// 	v.Children = make([]*Value, 0)
	// } else {
	// 	v.Children = v.Children[0:0]
	// }
	// v.Parents = make(map[int]*Value)

	// seen := make(map[*Value]bool)
	// for _, val := range children {
	// 	if !seen[val] {
	// 		v.Children = append(v.Children, val)
	// 		seen[val] = true
	// 	}
	// }

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
	t := NewValue(-v.Value, "neg")
	t.Children.Append(v)

	t.LocalBackward = func() {
		v.Grad += -1.0 * t.Grad
	}
	return t
}

func (v *Value) Add(other *Value) *Value {
	t := NewValue(v.Value+other.Value, "+")
	t.Children.Append(v)
	t.Children.Append(other)

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
	t := NewValue(v.Value*other.Value, "*")
	t.Children.Append(v)
	t.Children.Append(other)

	t.LocalBackward = func() {
		v.Grad += t.Grad * other.Value
		other.Grad += t.Grad * v.Value
	}

	return t
}

func (v *Value) Max(other *Value) *Value {
	t := NewValue(math.Max(v.Value, other.Value), "max")
	t.Children.Append(v)
	t.Children.Append(other)

	t.LocalBackward = func() {
		if v.Value > other.Value {
			v.Grad += t.Grad
			other.Grad += 0
		} else {
			v.Grad += 0
			other.Grad += t.Grad
		}
	}

	return t
}

func (v *Value) Div(other *Value) *Value {
	return v.Mul(other.Pow(-1))
}

func (v *Value) Log() *Value {
	epsilon := 1e-7
	safeValue := math.Max(v.Value, epsilon)
	t := NewValue(math.Log(safeValue), "log")
	t.Children.Append(v)

	t.LocalBackward = func() {
		v.Grad += (1.0 / safeValue) * t.Grad
	}
	return t
}

func (v *Value) Tanh() *Value {
	t := NewValue(math.Tanh(v.Value), "tanh")
	t.Children.Append(v)

	t.LocalBackward = func() {
		v.Grad += (1.0 - math.Pow(t.Value, 2.0)) * t.Grad
	}
	return t
}

func (v *Value) ReLu() *Value {
	t := NewValue(max(0, v.Value), "relu")
	t.Children.Append(v)

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
	t := NewValue(val, "sigmoid")
	t.Children.Append(v)

	t.LocalBackward = func() {
		v.Grad += (t.Value) * (1.0 - v.Value) * t.Grad
	}
	return t
}

func (v *Value) Exp() *Value {
	t := NewValue(math.Exp(v.Value), "exp")
	t.Children.Append(v)

	t.LocalBackward = func() {
		v.Grad += t.Value * t.Grad
	}
	return t
}

func (v *Value) Pow(other float64) *Value {
	t := NewValue(math.Pow(v.Value, other), "pow")
	t.Children.Append(v)

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
	for i := 0; i < root.Children.Len(); i++ {
		// for _, child := range root.Children {
		dfs(root.Children.At(i), visited, collect)
	}
	*collect = append(*collect, root)
}

func (v *Value) Backward() {
	visited := make(map[int]bool)
	collect := make([]*Value, 0)

	dfs(v, &visited, &collect)
	for i := range collect {
		collect[i].Grad = 0.0
	}
	v.Grad = 1.0
	for i := len(collect) - 1; i >= 0; i-- {
		collect[i].PerformBackward()
	}

	collect = nil
	visited = nil
}

func (v *Value) ClearOldChildren() {
	v.Children.Clear()
	v.LocalBackward = nil
}

func (v *Value) Clear() {
	v.ClearOldChildren()
	v.Grad = 0
}
