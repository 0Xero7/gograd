package lossfunctions

import (
	"gograd/ng"
	"math"
)

func TensorCrossEntropy(logits *ng.Tensor, classes *ng.Tensor) *ng.Tensor {
	return nil
}

// last dim of classes MUST be a probability distribution.
// Dims MUST be [Batch, Classes, d1, d2, ..., dk] (like in pytorch)
func TensorCrossEntropyProbDist(logits *ng.Tensor, classes *ng.Tensor) *ng.Tensor {
	N := logits.Shape[0]
	C := logits.Shape[1]

	probs := logits.SoftMax(1)

	// Compute cross entropy loss using one-hot encoded classes
	epsilon := 1e-15 // For numerical stability
	loss := 0.0
	sliceDims := make([]int, classes.Dim())

	for {
		for b := range N {
			sliceDims[0] = b
			lossItem := 0.0

			for c := range C {
				sliceDims[1] = c
				loss += -math.Log(epsilon+probs.Get(sliceDims...)) * classes.Get(sliceDims...)
			}

			loss += lossItem
		}

		if !ng.NextSlicedIndexMultiAxis(classes, &sliceDims, 0, 1) {
			break
		}
	}
	loss /= float64(logits.Len())
	loss *= float64(C)

	scaleFactor := float64(logits.Len() / C)

	// Create output tensor with proper backward pass
	out := ng.NewTensorFlatWith([]float64{loss}, []int{1}, "crossentropy", logits)

	out.LocalBackward = func() {
		for {
			for c := range C {
				sliceDims[1] = c
				accum := logits.GetGradient(sliceDims...) + (probs.Get(sliceDims...)-classes.Get(sliceDims...))/scaleFactor
				logits.SetGradient(accum, sliceDims...)
			}

			if !ng.NextSlicedIndexMultiAxis(logits, &sliceDims, 1) {
				break
			}
		}
	}

	return out
}
