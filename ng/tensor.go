package ng

import (
	"fmt"
	"gograd/utils"
	"math"
	"slices"
	"strings"
	"sync/atomic"
)

type Tensor struct {
	Id            int
	Value         []float64
	Grad          []float64
	Shape         []int
	Strides       []int
	LocalBackward func()
	Op            string
	Children      TensorChildren

	RequiresOptimization bool
}

var TidGen atomic.Int64

func NewTensorFlat(values []float64, shape []int) *Tensor {
	tensor := &Tensor{
		Id:       int(TidGen.Add(1)),
		Value:    make([]float64, len(values)),
		Grad:     make([]float64, len(values)),
		Shape:    make([]int, len(shape)),
		Strides:  make([]int, len(shape)),
		Op:       "None",
		Children: *NewTensorChildren(),
	}

	copy(tensor.Value, values)
	copy(tensor.Shape, shape)

	tensor.recalcuateStrides()

	return tensor
}

func NewTensorFlatWith(values []float64, shape []int, op string, children ...*Tensor) *Tensor {
	tensor := &Tensor{
		Id:       int(TidGen.Add(1)),
		Value:    make([]float64, len(values)),
		Grad:     make([]float64, len(values)),
		Shape:    make([]int, len(shape)),
		Strides:  make([]int, len(shape)),
		Op:       op,
		Children: *NewTensorChildrenWith(children),
	}

	copy(tensor.Value, values)
	copy(tensor.Shape, shape)

	tensor.recalcuateStrides()

	return tensor
}

func Scalar(v float64) *Tensor {
	return NewTensorFlat([]float64{v}, []int{1})
}

func Tensor1D(data []float64) *Tensor {
	return NewTensorFlat(data, []int{len(data)})
}

func Tensor2D(data [][]float64) *Tensor {
	rows := len(data)
	cols := len(data[0])
	flattened := make([]float64, rows*cols)
	for i := range rows {
		for j := range cols {
			flattened[rows*i+j] = data[i][j]
		}
	}
	return NewTensorFlat(flattened, []int{rows, cols})
}

func TensorOnes(shape ...int) *Tensor {
	size := shape[1]
	for _, w := range shape[1:] {
		size *= w
	}

	vals := make([]float64, size)
	for i := range vals {
		vals[i] = 1
	}

	return NewTensorFlat(vals, shape)
}

func TensorConst(value float64, shape ...int) *Tensor {
	size := shape[0]
	for _, w := range shape[1:] {
		size *= w
	}

	vals := make([]float64, size)
	for i := range vals {
		vals[i] = value
	}

	return NewTensorFlat(vals, shape)
}

// -----------------------------------------------------------------------------------------------------------------------

func (t *Tensor) Len() int {
	size := t.Shape[0]
	for _, w := range t.Shape[1:] {
		size *= w
	}
	return size
}

func (t *Tensor) recalcuateStrides() {
	stride := 1
	t.Strides = make([]int, len(t.Shape))
	for i := len(t.Shape) - 1; i >= 0; i-- {
		t.Strides[i] = stride
		stride = stride * t.Shape[i]
	}
}

func (t *Tensor) Dim() int {
	return len(t.Shape)
}

func (t *Tensor) String() string {
	// Helper function to build nested representation
	var build func([]int, int, int) string
	build = func(idx []int, dim int, stride int) string {
		if dim == len(t.Shape) {
			// Base case: get the actual value
			flatIdx := 0
			for i := 0; i < len(idx); i++ {
				flatIdx += idx[i] * t.Strides[i]
			}
			return fmt.Sprintf("id=%d, value=%.4f, grad=%.4f", t.Id, t.Value[flatIdx], t.Grad[flatIdx])
		}

		var sb strings.Builder
		sb.WriteString("[")

		for i := 0; i < t.Shape[dim]; i++ {
			idx[dim] = i
			if i > 0 {
				sb.WriteString(" ")
			}
			sb.WriteString(build(idx, dim+1, stride*t.Shape[dim]))
		}

		sb.WriteString("]")

		// Add newlines for better formatting of higher dimensions
		if dim < len(t.Shape)-2 {
			sb.WriteString("\n")
			// Add indentation
			for i := 0; i <= dim; i++ {
				sb.WriteString(" ")
			}
		}

		return sb.String()
	}

	return build(make([]int, len(t.Shape)), 0, 1)
}

// -----------------------------------------------------------------------------------------------------------------------

func (t *Tensor) Get(index ...int) float64 {
	if len(index) != len(t.Shape) {
		panic("Get index doesn't match tensor dimensions")
	}

	i := 0
	for j, stride := range t.Strides {
		i += index[j] * stride
	}

	return t.Value[i]
}

func (t *Tensor) Set(value float64, index ...int) {
	if len(index) != len(t.Shape) {
		panic("Get index doesn't match tensor dimensions")
	}

	i := 0
	for j, stride := range t.Strides {
		i += index[j] * stride
	}

	t.Value[i] = value
}

func (t *Tensor) Reshape(dims ...int) {
	existingSize := t.Len()
	newSize := dims[0]
	for _, w := range dims[1:] {
		newSize *= w
	}

	if newSize != existingSize {
		panic("Reshaping to an invalid size")
	}

	t.Shape = make([]int, len(dims))
	copy(t.Shape, dims)
	t.recalcuateStrides()
}

func (t *Tensor) Clone() *Tensor {
	tensor := NewTensorFlat(t.Value, t.Shape)
	tensor.Op = t.Op
	tensor.Children = *t.Children.Clone()
	copy(tensor.Grad, t.Grad)
	tensor.recalcuateStrides()
	return tensor
}

func (t *Tensor) CloneWith(op string, children ...*Tensor) *Tensor {
	tensor := NewTensorFlat(t.Value, t.Shape)
	tensor.Op = op
	tensor.Children = *NewTensorChildrenWith(children)
	copy(tensor.Grad, t.Grad)
	tensor.recalcuateStrides()
	return tensor
}

func (t *Tensor) Transpose(dim1, dim2 int) *Tensor {
	tensor := t.Clone()
	tensor.Shape[dim1], tensor.Shape[dim2] = tensor.Shape[dim2], tensor.Shape[dim1]
	tensor.Strides[dim1], tensor.Strides[dim2] = tensor.Strides[dim2], tensor.Strides[dim1]
	return tensor
}

func (t *Tensor) Extend(size int) {
	utils.AssertTrue(len(t.Shape) == 1 && t.Shape[0] == 1, "Cannot extend non 0 dimensional tensors yet")
	utils.AssertTrue(size > t.Len(), "New size is smaller than existing size")

	t.Shape[0] = size
	for len(t.Grad) < size {
		t.Value = append(t.Value, t.Value[0])
		t.Grad = append(t.Grad, t.Grad[0])
	}
}

// -----------------------------------------------------------------------------------------------------------------------

func (t *Tensor) Negate() *Tensor {
	vals := make([]float64, t.Len())
	for i := range t.Len() {
		vals[i] = -t.Value[i]
	}

	out := NewTensorFlatWith(vals, t.Shape, "negate", t)
	backward := func() {
		for i := range t.Len() {
			t.Grad[i] += -out.Grad[i]
		}
	}
	out.LocalBackward = backward

	return out
}

func (t *Tensor) Exp() *Tensor {
	vals := make([]float64, t.Len())
	for i := range t.Len() {
		vals[i] = math.Exp(t.Value[i])
	}

	out := NewTensorFlatWith(vals, t.Shape, "exp", t)
	backward := func() {
		for i := range t.Len() {
			t.Grad[i] += out.Value[i] * out.Grad[i]
		}
	}
	out.LocalBackward = backward

	return out
}

func (t *Tensor) Pow(other float64) *Tensor {
	vals := make([]float64, t.Len())
	for i := range t.Len() {
		vals[i] = math.Pow(t.Value[i], other)
	}

	out := NewTensorFlatWith(vals, t.Shape, "pow", t)
	backward := func() {
		for i := range t.Len() {
			t.Grad[i] += other * math.Pow(t.Value[i], other-1) * out.Grad[i]
		}
	}
	out.LocalBackward = backward

	return out
}

func (t *Tensor) Log() *Tensor {
	epsilon := 1e-7
	vals := make([]float64, t.Len())
	for i := range t.Len() {
		safeValue := math.Max(t.Value[i], epsilon)
		vals[i] = math.Log(safeValue)
	}

	out := NewTensorFlatWith(vals, t.Shape, "log", t)
	backward := func() {
		for i := range t.Len() {
			safeValue := math.Max(t.Value[i], epsilon)
			t.Grad[i] += (1.0 / safeValue) * out.Grad[i]
		}
	}
	out.LocalBackward = backward

	return out
}

func (t *Tensor) Tanh() *Tensor {
	vals := make([]float64, t.Len())
	for i := range t.Len() {
		vals[i] = math.Tanh(t.Value[i])
	}

	out := NewTensorFlatWith(vals, t.Shape, "tanh", t)
	backward := func() {
		for i := range t.Len() {
			t.Grad[i] += (1.0 - math.Pow(out.Value[i], 2.0)) * out.Grad[i]
		}
	}
	out.LocalBackward = backward

	return out
}

func (t *Tensor) ReLu() *Tensor {
	vals := make([]float64, t.Len())
	for i := range t.Len() {
		vals[i] = math.Max(0, t.Value[i])
	}

	out := NewTensorFlatWith(vals, t.Shape, "relu", t)

	backward := func() {
		for i := range t.Len() {
			grad := 0.0
			if out.Value[i] > 0 {
				grad = 1.0
			}

			t.Grad[i] += grad * out.Grad[i]
		}
	}
	out.LocalBackward = backward

	return out
}

func (t *Tensor) Sigmoid() *Tensor {
	vals := make([]float64, t.Len())
	for i := range t.Len() {
		vals[i] = 1.0 / (1.0 + math.Exp(-t.Value[i]))
	}

	out := NewTensorFlatWith(vals, t.Shape, "relu", t)

	backward := func() {
		for i := range t.Len() {
			t.Grad[i] += out.Value[i] * (1.0 - out.Value[i]) * out.Grad[i]
		}
	}
	out.LocalBackward = backward

	return out
}

func (t *Tensor) Sum() *Tensor {
	sum := 0.0
	for i := range t.Len() {
		sum += t.Value[i]
	}

	out := NewTensorFlatWith([]float64{sum}, []int{1}, "sum", t)

	backward := func() {
		for i := range t.Len() {
			t.Grad[i] += out.Grad[0]
		}
	}
	out.LocalBackward = backward

	return out
}

func (t *Tensor) MaximumElement() *Tensor {
	biggest := t.Value[0]
	for i := 1; i < t.Len(); i++ {
		if t.Value[i] > biggest {
			biggest = t.Value[i]
		}
	}

	out := NewTensorFlatWith([]float64{biggest}, []int{1}, "maximum_element", t)

	backward := func() {
		biggest := t.Value[0]
		biggestIndex := 0
		for i := 1; i < t.Len(); i++ {
			if t.Value[i] > biggest {
				biggest = t.Value[i]
				biggestIndex = i
			}
		}

		t.Grad[biggestIndex] += out.Grad[0]
	}
	out.LocalBackward = backward

	return out
}

func (t *Tensor) Choose(index int) *Tensor {
	out := NewTensorFlatWith([]float64{t.Value[index]}, []int{1}, "choose", t)

	backward := func() {
		// fmt.Println("Backprop called. Index =", index, ". Value =", out)
		t.Grad[index] += out.Grad[0]
	}
	out.LocalBackward = backward

	return out
}

// -----------------------------------------------------------------------------------------------------------------------

func (t *Tensor) Add(other *Tensor) *Tensor {
	utils.AssertTrue(slices.Equal(t.Shape, other.Shape), "Tensor shapes are not equal for <Add>")

	n := t.Len()
	vals := make([]float64, n)
	for i := range n {
		vals[i] = t.Value[i] + other.Value[i]
	}

	out := NewTensorFlatWith(vals, t.Shape, "add", t, other)
	backward := func() {
		grad := out.Grad
		tGrad := t.Grad
		otherGrad := other.Grad

		// Use SIMD-friendly loop
		for i := 0; i < n; i++ {
			g := grad[i]
			tGrad[i] += g
			otherGrad[i] += g
		}
	}
	out.LocalBackward = backward

	return out
}

func (t *Tensor) Sub(other *Tensor) *Tensor {
	utils.AssertTrue(slices.Equal(t.Shape, other.Shape), "Tensor shapes are not equal for <Sub>")

	return t.Add(other.Negate())
}

func (t *Tensor) Mul(other *Tensor) *Tensor {
	utils.AssertTrue(slices.Equal(t.Shape, other.Shape), "Tensor shapes are not equal for <Mul>")

	n := t.Len()
	vals := make([]float64, n)
	for i := range n {
		vals[i] = t.Value[i] * other.Value[i]
	}

	out := NewTensorFlatWith(vals, t.Shape, "mul", t, other)
	backward := func() {
		tGrad := t.Grad
		oGrad := other.Grad

		tVal := t.Value
		oVal := other.Value

		outGrad := out.Grad

		for i := range n {
			tGrad[i] += oVal[i] * outGrad[i]
			oGrad[i] += tVal[i] * outGrad[i]
		}
	}
	out.LocalBackward = backward

	return out
}

func (t *Tensor) Div(other *Tensor) *Tensor {
	utils.AssertTrue(slices.Equal(t.Shape, other.Shape), "Tensor shapes are not equal for <Div>")

	return t.Mul(other.Pow(-1))
}

func (t *Tensor) Max(other *Tensor) *Tensor {
	utils.AssertTrue(slices.Equal(t.Shape, other.Shape), "Tensor shapes are not equal for <Max>")

	vals := make([]float64, t.Len())
	for i := range t.Len() {
		vals[i] = math.Max(t.Value[i], other.Value[i])
	}

	out := NewTensorFlatWith(vals, t.Shape, "max", t, other)
	backward := func() {
		for i := range t.Len() {
			if t.Value[i] > other.Value[i] {
				t.Grad[i] += out.Grad[i]
			} else {
				other.Grad[i] += out.Grad[i]
			}
		}
	}
	out.LocalBackward = backward

	return out
}

// -----------------------------------------------------------------------------------------------------------------------

func dfsT(root *Tensor, visited *map[int]bool, collect *[]*Tensor) {
	if _, found := (*visited)[root.Id]; found {
		return
	}

	(*visited)[root.Id] = true
	for i := 0; i < root.Children.Len(); i++ {
		dfsT(root.Children.At(i), visited, collect)
	}
	*collect = append(*collect, root)
}

var PathT []*Tensor = make([]*Tensor, 0)

func (t *Tensor) PerformBackward() {
	if t.LocalBackward != nil {
		t.LocalBackward()
	}
}

func (t *Tensor) Backward() {
	// if len(PathT) == 0 {
	visited := make(map[int]bool)
	collect := make([]*Tensor, 0)
	dfsT(t, &visited, &collect)
	PathT = make([]*Tensor, 0)
	for i := len(collect) - 1; i >= 0; i-- {
		PathT = append(PathT, collect[i])
	}
	// }

	for i := range PathT {
		for j := range PathT[i].Grad {
			PathT[i].Grad[j] = 0.0
		}
	}
	for i := range t.Grad {
		t.Grad[i] = 1.0
	}

	for i := range PathT {
		PathT[i].PerformBackward()
	}
}
