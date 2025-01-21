package ng

import (
	"fmt"
	"gograd/utils"
	"math"
	"math/rand"
	"slices"
	"strings"
	"sync/atomic"

	"github.com/kelindar/simd"
	"gonum.org/v1/gonum/mat"
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

	Size                 int
	RequiresOptimization bool
}

var TidGen atomic.Int64
var TTensorPool ValuePool[Tensor] = NewValuePool(func(index int) *Tensor {
	r := new(Tensor)
	r.Id = index
	r.Children.Clear()
	return r
})

func NewTensorFlat(values []float64, shape []int) *Tensor {
	tensor, exists := TTensorPool.Get()
	if !exists {
		tensor.Value = make([]float64, len(values))
		tensor.Grad = make([]float64, len(values))
		tensor.Shape = make([]int, len(shape))
		tensor.Strides = make([]int, len(shape))
		tensor.Op = "None"
		tensor.Children = *NewTensorChildren()
	} else {
		tensor.Children.Clear()
	}

	copy(tensor.Value, values)
	copy(tensor.Shape, shape)

	tensor.recalcuateStrides()

	return tensor
}

func NewTensorFlatWith(values []float64, shape []int, op string, children ...*Tensor) *Tensor {
	tensor, exists := TTensorPool.Get()
	// fmt.Println(tensor.Id)
	if !exists {
		tensor.Value = make([]float64, len(values))
		tensor.Grad = make([]float64, len(values))
		tensor.Shape = make([]int, len(shape))
		tensor.Strides = make([]int, len(shape))
		tensor.Op = op
		tensor.Children = *NewTensorChildrenWith(children)
	} else {
		tensor.Children.Clear()
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
			flattened[cols*i+j] = data[i][j]
		}
	}
	return NewTensorFlat(flattened, []int{rows, cols})
}

func TensorOnes(shape ...int) *Tensor {
	size := shape[0]
	if len(shape) > 1 {
		for _, w := range shape[1:] {
			size *= w
		}
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

func TensorRand(shape ...int) *Tensor {
	size := shape[0]
	for _, w := range shape[1:] {
		size *= w
	}

	vals := make([]float64, size)
	for i := range vals {
		vals[i] = rand.NormFloat64()
	}

	return NewTensorFlat(vals, shape)
}

// -----------------------------------------------------------------------------------------------------------------------

func (t *Tensor) Len() int {
	return t.Size
}

func (t *Tensor) recalcuateStrides() {
	stride := 1
	t.Strides = make([]int, len(t.Shape))
	for i := len(t.Shape) - 1; i >= 0; i-- {
		t.Strides[i] = stride
		stride = stride * t.Shape[i]
	}
	t.Size = stride
}

func (t *Tensor) Dim() int {
	return len(t.Shape)
}

func (t *Tensor) String() string {
	// Helper function to build nested representation
	var build func([]int, int, int) string
	build = func(idx []int, axis int, stride int) string {
		if axis == len(t.Shape) {
			// Base case: get the actual value
			flatIdx := 0
			for i := 0; i < len(idx); i++ {
				flatIdx += idx[i] * t.Strides[i]
			}
			// return fmt.Sprintf("id=%d, value=%.4f, grad=%.4f", t.Id, t.Value[flatIdx], t.Grad[flatIdx])
			return fmt.Sprintf("%.4f, ", t.Value[flatIdx])
		}

		var sb strings.Builder
		sb.WriteString("[")

		for i := 0; i < t.Shape[axis]; i++ {
			idx[axis] = i
			if i > 0 {
				sb.WriteString(" ")
			}
			sb.WriteString(build(idx, axis+1, stride*t.Shape[axis]))
		}

		sb.WriteString("],\n")

		// Add newlines for better formatting of higher dimensions
		if axis < len(t.Shape)-2 {
			sb.WriteString("\n")
			// Add indentation
			for i := 0; i <= axis; i++ {
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

func (t *Tensor) GetGradient(index ...int) float64 {
	if len(index) != len(t.Shape) {
		panic("Get index doesn't match tensor dimensions")
	}

	i := 0
	for j, stride := range t.Strides {
		i += index[j] * stride
	}

	return t.Grad[i]
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

func (t *Tensor) SetGradient(value float64, index ...int) {
	if len(index) != len(t.Shape) {
		panic("Get index doesn't match tensor dimensions")
	}

	i := 0
	for j, stride := range t.Strides {
		i += index[j] * stride
	}

	t.Grad[i] = value
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
	tensor.Id = int(TidGen.Add(1))
	tensor.Op = t.Op
	tensor.Children = *t.Children.Clone()
	copy(tensor.Grad, t.Grad)
	tensor.recalcuateStrides()
	return tensor
}

func (t *Tensor) CloneWith(op string, children ...*Tensor) *Tensor {
	tensor := NewTensorFlat(t.Value, t.Shape)
	tensor.Id = int(TidGen.Add(1))
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
		x := t.Value[i]
		if x > 5 {
			vals[i] = 1.0
		} else if x < -5 {
			vals[i] = -1.0
		} else {
			vals[i] = math.Tanh(x)
		}
	}

	out := NewTensorFlatWith(vals, t.Shape, "tanh", t)
	backward := func() {
		for i := range t.Len() {
			y := out.Value[i]
			y = y * y
			t.Grad[i] += (1.0 - y) * out.Grad[i]
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

func (t *Tensor) SoftMax(axis int) *Tensor {
	axis = (axis + t.Dim()) % t.Dim()

	out := t.CloneWith("softmax", t)

	sliceIndices := make([]int, t.Dim())
	for {
		dimSize := t.Shape[axis]
		biggest := math.Inf(-1)

		logits := make([]float64, dimSize)
		for i := range dimSize {
			sliceIndices[axis] = i
			elem := t.Get(sliceIndices...)
			logits[i] = elem
			// fmt.Println(sliceIndices, elem)

			if elem > biggest {
				biggest = elem
			}
		}

		shiftedSum := 0.0
		for i := range dimSize {
			logits[i] = math.Exp(logits[i] - biggest)
			shiftedSum += logits[i]
		}

		for i := range dimSize {
			logits[i] /= shiftedSum
			sliceIndices[axis] = i
			out.Set(logits[i], sliceIndices...)
		}

		if !NextSlicedIndex(t, axis, &sliceIndices) {
			break
		}
	}

	out.LocalBackward = func() {
		// copy(t.Grad, out.Grad)

		s := out.Value
		for i := range t.Grad {
			for j := range out.Grad {
				if i == j {
					t.Grad[i] += out.Grad[j] * s[i] * (1 - s[i])
				} else {
					t.Grad[i] += out.Grad[j] * (-s[i] * s[j])
				}
			}
		}
	}

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

func (t *Tensor) ReshapeOut(dims ...int) *Tensor {
	out := t.Clone()
	out.Reshape(dims...)
	out.Children.Clear()
	out.Children.Append(t)

	out.Op = "reshape"
	out.LocalBackward = func() {
		for i := range t.Len() {
			t.Grad[i] += out.Grad[i]
		}
	}

	return out
}

// -----------------------------------------------------------------------------------------------------------------------

func fastMatrixMultiply(a, b *Tensor) []float64 {
	c := mat.NewDense(a.Shape[0], a.Shape[1], a.Value)
	d := mat.NewDense(b.Shape[0], b.Shape[1], b.Value)
	var res mat.Dense
	res.Mul(c, d)
	return res.RawMatrix().Data
}

func fastMatrixMultiplyRaw(a, b mat.Matrix) []float64 {
	var res mat.Dense
	res.Mul(a, b)
	return res.RawMatrix().Data
}

func (t *Tensor) MatMul(other *Tensor) *Tensor {
	utils.AssertTrue(t.Dim() == 2 && other.Dim() == 2, "Can't perform matmul on operands that are not 2D.")
	utils.AssertTrue(t.Shape[1] == other.Shape[0], fmt.Sprint("Can't perform matmul on matrices of shape ", t.Shape, " and ", other.Shape))

	out := NewTensorFlat(fastMatrixMultiply(t, other), []int{t.Shape[0], other.Shape[1]})

	out.Children.Append(t)
	out.Children.Append(other)
	out.Op = "matmul"
	out.LocalBackward = func() {
		dc := mat.NewDense(out.Shape[0], out.Shape[1], out.Grad)
		at := mat.NewDense(t.Shape[0], t.Shape[1], t.Value)
		bt := mat.NewDense(other.Shape[0], other.Shape[1], other.Value)

		a_grad := fastMatrixMultiplyRaw(dc, bt.T())
		b_grad := fastMatrixMultiplyRaw(at.T(), dc)

		simd.AddFloat64s(t.Grad, t.Grad, a_grad)
		simd.AddFloat64s(other.Grad, other.Grad, b_grad)
	}

	return out
}

// -----------------------------------------------------------------------------------------------------------------------

func (t *Tensor) Add(other *Tensor) *Tensor {
	addBackward := func(thisValue, thisGrad, otherValue, otherGrad, outValue, outGrad float64) (float64, float64) {
		return outGrad, outGrad
	}

	addForward := func(thisValue, otherValue float64) float64 {
		return thisValue + otherValue
	}

	return PerformBinaryOp(
		"add",
		t, other,

		simd.MulFloat64s,
		addBackward,

		addForward,
		addBackward,
	)
}

func (t *Tensor) Sub(other *Tensor) *Tensor {
	return t.Add(other.Negate())
}

func (t *Tensor) Mul(other *Tensor) *Tensor {
	mulBackward := func(thisValue, thisGrad, otherValue, otherGrad, outValue, outGrad float64) (float64, float64) {
		return outGrad * otherValue, outGrad * thisValue
	}

	mulForward := func(thisValue, otherValue float64) float64 {
		return thisValue * otherValue
	}

	return PerformBinaryOp(
		"mul",
		t, other,

		simd.MulFloat64s,
		mulBackward,

		mulForward,
		mulBackward,
	)
}

func (t *Tensor) Div(other *Tensor) *Tensor {
	utils.AssertTrue(slices.Equal(t.Shape, other.Shape), "Tensor shapes are not equal for <Div>")

	return t.Mul(other.Pow(-1))
}

func (t *Tensor) Max(other *Tensor) *Tensor {
	maxFowardFast := func(dst, this, other []float64) []float64 {
		for i := range dst {
			dst[i] = math.Max(this[i], other[i])
		}
		return dst
	}

	maxForward := func(thisValue, otherValue float64) float64 {
		return math.Max(thisValue, otherValue)
	}

	maxBackward := func(thisValue, thisGrad, otherValue, otherGrad, outValue, outGrad float64) (float64, float64) {
		if thisValue > otherValue {
			return outGrad, 0
		} else {
			return 0, outGrad
		}
	}

	return PerformBinaryOp(
		"max",
		t, other,

		maxFowardFast,
		maxBackward,
		maxForward,
		maxBackward,
	)
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

func (t *Tensor) Backward(init ...Tensor) {
	// if len(PathT) == 0 {
	visited := make(map[int]bool)
	collect := make([]*Tensor, 0)
	dfsT(t, &visited, &collect)
	PathT = make([]*Tensor, 0)
	for i := len(collect) - 1; i >= 0; i-- {
		PathT = append(PathT, collect[i])
	}
	// }

	pathLen := len(PathT)
	for i := range pathLen {
		grads := PathT[i].Grad
		n := len(grads)
		for j := range n {
			grads[j] = 0.0
		}
	}

	tGrad := t.Grad
	tLen := len(tGrad)

	if len(init) != 0 {
		copy(t.Grad, init[0].Value)
	} else {
		for i := range tLen {
			tGrad[i] = 1.0
		}
	}

	for i := range pathLen {
		PathT[i].PerformBackward()
	}
}
