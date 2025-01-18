package ng

import (
	"fmt"
	"slices"
)

func CanBroadcast(t1, t2 *Tensor) bool {
	s1 := t1.Shape
	s2 := t2.Shape

	if (len(s1) == 0 || len(s2) == 0) && len(s1) != len(s2) {
		return false
	}

	m := min(len(s1), len(s2))
	for i := range m {
		a := s1[len(s1)-1-i]
		b := s2[len(s2)-1-i]

		if (a == b) || (a == 1 || b == 1) {
			continue
		}
		return false
	}
	return true
}

// GetBroadcastDims returns the output shape and expanded strides for broadcasting
func GetBroadcastDims(t1 *Tensor, t2 *Tensor) ([]int, []int, []int) {
	if !CanBroadcast(t1, t2) {
		panic(fmt.Sprintf("Cannot broadcast tensors with shapes %v and %v", t1.Shape, t2.Shape))
	}

	// Calculate output shape
	outShape := make([]int, max(len(t1.Shape), len(t2.Shape)))
	for i := range outShape {
		t1Idx := len(t1.Shape) - 1 - i
		t2Idx := len(t2.Shape) - 1 - i

		var dim1, dim2 int = 1, 1
		if t1Idx >= 0 {
			dim1 = t1.Shape[t1Idx]
		}
		if t2Idx >= 0 {
			dim2 = t2.Shape[t2Idx]
		}
		outShape[len(outShape)-1-i] = max(dim1, dim2)
	}

	// Create broadcasted view strides
	t1Strides := make([]int, len(outShape))
	t2Strides := make([]int, len(outShape))

	// Fill in strides from right to left
	for i := range outShape {
		origIdx1 := len(t1.Shape) - 1 - i
		origIdx2 := len(t2.Shape) - 1 - i

		if origIdx1 >= 0 {
			if t1.Shape[origIdx1] == outShape[len(outShape)-1-i] {
				t1Strides[len(outShape)-1-i] = t1.Strides[origIdx1]
			}
			// else stride is 0 for broadcast
		}

		if origIdx2 >= 0 {
			if t2.Shape[origIdx2] == outShape[len(outShape)-1-i] {
				t2Strides[len(outShape)-1-i] = t2.Strides[origIdx2]
			}
			// else stride is 0 for broadcast
		}
	}

	return outShape, t1Strides, t2Strides
}

// GetBroadcastIndices calculates the source indices for broadcasted tensors
func GetBroadcastIndices(flatIndex int, outShape []int, t1Strides []int, t2Strides []int) (int, int) {
	idx1, idx2 := 0, 0
	remain := flatIndex

	for j := len(outShape) - 1; j >= 0; j-- {
		dim := outShape[j]
		idx := remain % dim
		remain = remain / dim
		idx1 += idx * t1Strides[j]
		idx2 += idx * t2Strides[j]
	}

	return idx1, idx2
}

// Generic Binary Op Kernel
func PerformBinaryOp(
	op string,
	this, other *Tensor,

	fastPathForward func(dst, this, other []float64) []float64,
	fastPathBackward func(thisValue, thisGrad, otherValue, otherGrad, outValue, outGrad float64) (float64, float64),

	forward func(thisValue, otherValue float64) float64,
	backward func(thisValue, thisGrad, otherValue, otherGrad, outValue, outGrad float64) (float64, float64),
) *Tensor {
	// Fast path, no broadcasting
	if slices.Equal(this.Shape, other.Shape) {
		n := this.Len()
		vals := make([]float64, n)
		fastPathForward(vals, this.Value, other.Value)

		out := NewTensorFlatWith(vals, this.Shape, op, this, other)
		backward := func() {
			for i := range n {
				thisDelta, otherDelta := fastPathBackward(
					this.Value[i], this.Grad[i],
					other.Value[i], other.Grad[i],
					out.Value[i], out.Grad[i],
				)

				this.Grad[i] += thisDelta
				other.Grad[i] += otherDelta
			}
		}
		out.LocalBackward = backward
		return out
	}

	// Slow path, needs broadcasting

	// Get broadcast dimensions
	outShape, t1Strides, t2Strides := GetBroadcastDims(this, other)

	// Calculate total size
	totalSize := 1
	for _, dim := range outShape {
		totalSize *= dim
	}

	// Perform addition with broadcasting
	vals := make([]float64, totalSize)
	for i := 0; i < totalSize; i++ {
		idx1, idx2 := GetBroadcastIndices(i, outShape, t1Strides, t2Strides)
		vals[i] = forward(this.Value[idx1], other.Value[idx2])
		// vals[i] = this.Value[idx1] + other.Value[idx2]
	}

	out := NewTensorFlatWith(vals, outShape, op, this, other)

	// Backward pass uses the same broadcast indices
	out.LocalBackward = func() {
		for i := 0; i < totalSize; i++ {
			idx1, idx2 := GetBroadcastIndices(i, outShape, t1Strides, t2Strides)
			thisDelta, otherDelta := fastPathBackward(
				this.Value[idx1], this.Grad[idx1],
				other.Value[idx2], other.Grad[idx2],
				out.Value[i], out.Grad[i],
			)

			this.Grad[idx1] += thisDelta
			other.Grad[idx2] += otherDelta
			// this.Grad[idx1] += out.Grad[i]
			// other.Grad[idx2] += out.Grad[i]
		}
	}

	return out
}

func NextSlicedIndex(t *Tensor, axis int, indices *[]int) bool {
	hasNext := false
	for i := t.Dim() - 1; i >= 0; i-- {
		if axis == i {
			continue
		}

		(*indices)[i]++
		if (*indices)[i] == t.Shape[i] {
			(*indices)[i] = 0
			continue
		}

		hasNext = true
		break
	}

	if !hasNext {
		(*indices)[axis] = 0
	}

	return hasNext
}

// func IterateOverSlice(t *Tensor, axis int, iterator func(flatIndices *[][]int)) {
// 	slicedIndices := make([]int, t.Dim())
// 	for {
// 		for index := range t.Shape[axis] {
// 			iterator(index, slicedIndices...)
// 		}

// 		if !NextSlicedIndex(t, axis, &slicedIndices) {
// 			break
// 		}
// 	}
// }
