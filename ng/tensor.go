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
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
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

	firstTime bool
	cachedOut []float64
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
		// tensor.Value = make([]float64, len(values))
		tensor.Grad = make([]float64, len(values))
		// tensor.Shape = make([]int, len(shape))
		tensor.Strides = make([]int, len(shape))
		tensor.Op = "None"
		tensor.Children = *NewTensorChildren()
	} else {
		tensor.Children.Clear()
	}

	tensor.Value = values
	tensor.Shape = shape
	// copy(tensor.Value, values)
	// copy(tensor.Shape, shape)

	tensor.recalcuateStrides()

	return tensor
}

func NewTensorFlatWith(values []float64, shape []int, op string, children ...*Tensor) *Tensor {
	tensor, exists := TTensorPool.Get()
	// fmt.Println(tensor.Id)
	if !exists {
		// tensor.Value = make([]float64, len(values))
		tensor.Grad = make([]float64, len(values))
		// tensor.Shape = make([]int, len(shape))
		tensor.Strides = make([]int, len(shape))
		tensor.Op = op
		tensor.Children = *NewTensorChildrenWith(children)
	} else {
		tensor.Children.Clear()
	}

	tensor.Value = values
	tensor.Shape = shape
	// copy(tensor.Value, values)
	// copy(tensor.Shape, shape)

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

func (t *Tensor) Tanh3() *Tensor {
	// Reuse cached memory for outputs
	if cap(t.cachedOut) < t.Len() {
		t.cachedOut = make([]float64, t.Len())
	} else {
		t.cachedOut = t.cachedOut[:t.Len()]
	}

	// Vectorized forward pass using SIMD-friendly operations
	values := t.Value
	output := t.cachedOut
	for i := range values {
		x := values[i]
		// Use optimized math library implementation
		output[i] = math.Tanh(x)
	}

	out := NewTensorFlatWith(output, t.Shape, "tanh", t)

	// Optimized backward pass using precomputed values
	out.LocalBackward = func() {
		y := out.Value   // tanh(x)
		grad := out.Grad // ∂L/∂tanh(x)
		dst := t.Grad    // ∂L/∂x

		// Vectorized: dst += (1 - y²) * grad
		simdRange(len(y), func(i, n int) {
			// Compute (1 - y²) using SIMD multiplication
			ySlice := y[i : i+n]
			gradSlice := grad[i : i+n]
			dstSlice := dst[i : i+n]
			// subSlice := make([]float64, n)
			// for j := range n {
			// 	subSlice[j] = 1.0
			// }

			// temp = y * y
			temp := make([]float64, n)
			simd.MulFloat64s(temp, ySlice, ySlice)

			// temp = 1 - temp
			// simd.SubFloat64s(temp, simd.Broadcast(1.0, n), temp)
			// simd.SubFloat64s(temp, subSlice, temp)

			// temp *= grad
			simd.MulFloat64s(temp, temp, gradSlice)

			// Accumulate gradients
			simd.AddFloat64s(dstSlice, dstSlice, temp)
		})
	}

	return out
}

// Helper for SIMD-friendly chunk processing
func simdRange(n int, fn func(start, chunk int)) {
	const simdSize = 8 // AVX2 (256-bit) can process 4 doubles
	for i := 0; i < n; i += simdSize {
		chunk := min(simdSize, n-i)
		fn(i, chunk)
	}
}

func (t *Tensor) Tanh2() *Tensor {
	if !t.firstTime {
		t.cachedOut = make([]float64, t.Len())
		t.firstTime = true
	}

	for i := range t.Len() {
		x := t.Value[i]
		t.cachedOut[i] = 1 - (2 * (1 / (1 + math.Exp(x*2))))
		// if x > 5 {
		// 	vals[i] = 1.0
		// } else if x < -5 {
		// 	vals[i] = -1.0
		// } else {
		// 	vals[i] = math.Tanh(x)
		// }
	}

	out := NewTensorFlatWith(t.cachedOut, t.Shape, "tanh", t)
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

func (t *Tensor) Tanh() *Tensor {
	if !t.firstTime {
		t.cachedOut = make([]float64, t.Len())
		t.firstTime = true
	}

	for i := range t.Len() {
		x := t.Value[i]
		// t.cachedOut[i] = 1 - (2 * (1 / (1 + math.Exp(x*2))))
		if x > 5 {
			t.cachedOut[i] = 1.0
		} else if x < -5 {
			t.cachedOut[i] = -1.0
		} else {
			x2 := x * x
			a := x * (135135.0 + x2*(17325.0+x2*(378.0+x2)))
			b := 135135.0 + x2*(62370.0+x2*(3150.0+x2*28.0))
			t.cachedOut[i] = a / b
		}
	}

	out := NewTensorFlatWith(t.cachedOut, t.Shape, "tanh", t)
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

// func fastMatrixMultiply(a, b *Tensor) []float64 {
// 	c := mat.NewDense(a.Shape[0], a.Shape[1], a.Value)
// 	d := mat.NewDense(b.Shape[0], b.Shape[1], b.Value)
// 	var res mat.Dense
// 	res.Mul(c, d)
// 	return res.RawMatrix().Data
// }

var BlasImpl = blas64.Implementation()
var _pool [][]float64
var _poolIndex = 0

func ResetPoolIndex() {
	_poolIndex = 0
}

func fastMatrixMultiply(a, b *Tensor) []float64 {
	rowsA := a.Shape[0]
	colsA := a.Shape[1]
	colsB := b.Shape[1]
	colsOut := colsB // Output columns

	if len(a.cachedOut) != rowsA*colsOut {
		if len(a.cachedOut) < rowsA*colsOut {
			a.cachedOut = slices.Grow(a.cachedOut, rowsA*colsOut)
			for i := len(a.cachedOut); i < rowsA*colsOut; i++ {
				a.cachedOut = append(a.cachedOut, 0.0)
			}
		} else {
			a.cachedOut = a.cachedOut[:rowsA*colsOut]
		}
	}

	BlasImpl.Dgemm(
		blas.NoTrans, // Transpose A? (No)
		blas.NoTrans, // Transpose B? (No)
		rowsA,        // M: Number of rows of C and A
		colsOut,      // N: Number of columns of C and B
		colsA,        // K: Number of columns of A and rows of B
		1.0,          // alpha: scalar multiplier for A*B
		a.Value,      // A: Matrix A (data slice)
		colsA,        // lda: Leading dimension of A (number of columns of A, stride between rows)
		b.Value,      // B: Matrix B (data slice)
		colsB,        // ldb: Leading dimension of B (number of columns of B, stride between rows)
		0.0,          // beta: scalar multiplier for C (we want to overwrite C, so 0)
		a.cachedOut,  // C: Matrix C (data slice) - output
		colsOut,      // ldc: Leading dimension of C (number of columns of C, stride between rows)
	)

	return a.cachedOut
}

func (t *Tensor) MatMul(other *Tensor) *Tensor {
	utils.AssertTrue(t.Dim() == 2 && other.Dim() == 2, "Can't perform matmul on operands that are not 2D.")
	utils.AssertTrue(t.Shape[1] == other.Shape[0], fmt.Sprint("Can't perform matmul on matrices of shape ", t.Shape, " and ", other.Shape))

	out := NewTensorFlat(fastMatrixMultiply(t, other), []int{t.Shape[0], other.Shape[1]})

	out.Children.Append(t)
	out.Children.Append(other)
	out.Op = "matmul"
	out.LocalBackward = func() {
		rowsA := t.Shape[0]
		colsA := t.Shape[1]
		colsB := other.Shape[1]
		colsOut := out.Shape[1]

		BlasImpl.Dgemm(
			blas.NoTrans, // Transpose dc? No
			blas.Trans,   // Transpose bt? Yes (other.Value is B, we need B^T)
			rowsA,        // rows in a_grad (same as rows in dc, rows in A)
			colsA,        // cols in a_grad (same as cols in bt.T(), cols in B)
			colsOut,      // inner dimension (cols in dc, rows in bt)
			1.0,          // alpha
			out.Grad,     // matrix dc (out.Grad)
			colsOut,      // lda = cols of dc
			other.Value,  // matrix bt (other.Value)
			colsB,        // ldb = cols of bt (original B, not transposed in terms of leading dimension)
			1.0,          // beta (accumulate into t.Grad)
			t.Grad,       // matrix to accumulate to (t.Grad)
			colsA,        // ldc = cols of a_grad (cols of A)
		)

		BlasImpl.Dgemm(
			blas.Trans,   // Transpose at? Yes (t.Value is A, we need A^T)
			blas.NoTrans, // Transpose dc? No
			colsA,        // rows in b_grad (same as rows in at.T(), cols in A)
			colsB,        // cols in b_grad (same as cols in dc, cols in C, cols in B)
			rowsA,        // inner dimension (cols in at, rows in dc, rows in A)
			1.0,          // alpha
			t.Value,      // matrix at (t.Value)
			colsA,        // lda = cols of at (original A, not transposed in terms of leading dimension)
			out.Grad,     // matrix dc (out.Grad)
			colsOut,      // ldb = cols of dc
			1.0,          // beta (accumulate into other.Grad)
			other.Grad,   // matrix to accumulate to (other.Grad)
			colsB,        // ldc = cols of b_grad (cols of B)
		)
	}

	return out
}

// -----------------------------------------------------------------------------------------------------------------------

func (t *Tensor) Add2(other *Tensor) *Tensor {
	addBackward := func(thisValue, thisGrad, otherValue, otherGrad, outValue, outGrad float64) (float64, float64) {
		return outGrad, outGrad
	}

	addForward := func(thisValue, otherValue float64) float64 {
		return thisValue + otherValue
	}

	return PerformBinaryOp(
		"add",
		t, other,

		simd.AddFloat64s,
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

func BroadcastShapes(aShape, bShape []int) ([]int, error) {
	maxLen := len(aShape)
	if len(bShape) > maxLen {
		maxLen = len(bShape)
	}

	aPad := make([]int, maxLen)
	bPad := make([]int, maxLen)

	for i := 0; i < maxLen; i++ {
		aIdx := len(aShape) - maxLen + i
		if aIdx >= 0 {
			aPad[i] = aShape[aIdx]
		} else {
			aPad[i] = 1
		}

		bIdx := len(bShape) - maxLen + i
		if bIdx >= 0 {
			bPad[i] = bShape[bIdx]
		} else {
			bPad[i] = 1
		}
	}

	result := make([]int, maxLen)
	for i := 0; i < maxLen; i++ {
		aDim, bDim := aPad[i], bPad[i]
		if aDim != bDim && aDim != 1 && bDim != 1 {
			return nil, fmt.Errorf("shapes %v and %v are not broadcastable", aShape, bShape)
		}
		result[i] = max(aDim, bDim)
	}
	return result, nil
}

// func max(a, b int) int {
// 	if a > b {
// 		return a
// 	}
// 	return b
// }

func (t *Tensor) BroadcastTo(targetShape []int) (*Tensor, error) {
	bcShape, err := BroadcastShapes(t.Shape, targetShape)
	if err != nil {
		return nil, err
	}
	if !slices.Equal(bcShape, targetShape) {
		return nil, fmt.Errorf("cannot broadcast shape %v to %v", t.Shape, targetShape)
	}

	originalDims := len(t.Shape)
	alignedShape := make([]int, len(targetShape))
	for i := range alignedShape {
		if i < len(targetShape)-originalDims {
			alignedShape[i] = 1
		} else {
			alignedShape[i] = t.Shape[i-(len(targetShape)-originalDims)]
		}
	}

	newSize := product(targetShape)
	newData := make([]float64, newSize)
	originalStrides := makeStrides(alignedShape)

	indices := make([]int, len(targetShape))
	for i := 0; i < newSize; i++ {
		pos := i
		for j := len(targetShape) - 1; j >= 0; j-- {
			indices[j] = pos % targetShape[j]
			pos /= targetShape[j]
		}

		originalIndex := 0
		for j, idx := range indices {
			dim := alignedShape[j]
			if dim == 1 {
				idx = 0
			}
			originalIndex += idx * originalStrides[j]
		}
		newData[i] = t.Value[originalIndex]
	}

	out := NewTensorFlat(newData, targetShape)
	out.Op = "broadcast"
	out.Children.Append(t)

	out.LocalBackward = func() {
		grad := out.Grad
		for i := 0; i < len(grad); i++ {
			pos := i
			for j := len(targetShape) - 1; j >= 0; j-- {
				indices[j] = pos % targetShape[j]
				pos /= targetShape[j]
			}

			originalIndex := 0
			for j, idx := range indices {
				dim := alignedShape[j]
				if dim == 1 {
					idx = 0
				}
				originalIndex += idx * originalStrides[j]
			}
			t.Grad[originalIndex] += grad[i]
		}
	}

	return out, nil
}

func product(shape []int) int {
	p := 1
	for _, dim := range shape {
		p *= dim
	}
	return p
}

func makeStrides(shape []int) []int {
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

func (t *Tensor) Add(other *Tensor) *Tensor {
	addForward := func(dst, a, b []float64) []float64 {
		return simd.AddFloat64s(dst, a, b)
	}
	addBackward := func(_, aGrad, _, bGrad, _, outGrad float64) (float64, float64) {
		return aGrad + outGrad, bGrad + outGrad
	}
	return PerformBinaryOp("add", t, other, addForward, addBackward,
		func(a, b float64) float64 { return a + b },
		func(_, _, _, _, _, outGrad float64) (float64, float64) { return outGrad, outGrad },
	)
}

func (t *Tensor) addUnbroadcasted(other *Tensor) *Tensor {
	addBackward := func(tVal, tGrad, oVal, oGrad, outVal, outGrad float64) (float64, float64) {
		return outGrad, outGrad
	}
	addForward := func(tVal, oVal float64) float64 {
		return tVal + oVal
	}
	return PerformBinaryOp("add", t, other, simd.AddFloat64s, addBackward, addForward, addBackward)
}
