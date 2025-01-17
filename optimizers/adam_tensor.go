package optimizers

import (
	"gograd/ng"
	"math"
)

type AdamTensor struct {
	lr           float64
	beta1, beta2 float64
	epsilon      float64
	m, v         map[*ng.Tensor][]float64
	t            int
}

func NewAdamTensor(lr float64) *AdamTensor {
	return &AdamTensor{
		lr:      lr,
		beta1:   0.9,   // Exponential decay rate for first moment
		beta2:   0.999, // Exponential decay rate for second moment
		epsilon: 1e-8,  // Small constant to prevent division by zero
		m:       make(map[*ng.Tensor][]float64),
		v:       make(map[*ng.Tensor][]float64),
		t:       0,
	}
}

func (a *AdamTensor) Step(params []*ng.Tensor) {
	a.t++

	for _, param := range params {
		if param == nil || !param.RequiresOptimization {
			continue
		}

		// grad := param.Grad

		// Initialize momentum and velocity if not present
		if _, exists := a.m[param]; !exists {
			a.m[param] = make([]float64, param.Len())
			a.v[param] = make([]float64, param.Len())
		}

		for index := range param.Len() {
			// Update biased first moment estimate
			a.m[param][index] = a.beta1*a.m[param][index] + (1-a.beta1)*param.Grad[index]

			// Update biased second moment estimate
			a.v[param][index] = a.beta2*a.v[param][index] + (1-a.beta2)*param.Grad[index]*param.Grad[index]

			// Compute bias-corrected first moment estimate
			mHat := a.m[param][index] / (1 - math.Pow(a.beta1, float64(a.t)))

			// Compute bias-corrected second moment estimate
			vHat := a.v[param][index] / (1 - math.Pow(a.beta2, float64(a.t)))

			// Update parameters
			param.Value[index] -= a.lr * mHat / (math.Sqrt(vHat) + a.epsilon)
		}
	}
}
