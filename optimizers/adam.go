package optimizers

import (
	"gograd/ng"
	"math"
)

type Adam struct {
	lr           float64
	beta1, beta2 float64
	epsilon      float64
	m, v         map[*ng.Value]float64
	t            int
}

func NewAdam(lr float64) *Adam {
	return &Adam{
		lr:      lr,
		beta1:   0.9,   // Exponential decay rate for first moment
		beta2:   0.999, // Exponential decay rate for second moment
		epsilon: 1e-8,  // Small constant to prevent division by zero
		m:       make(map[*ng.Value]float64),
		v:       make(map[*ng.Value]float64),
		t:       0,
	}
}

func (a *Adam) Step(params []*ng.Value) {
	a.t++

	for _, param := range params {
		if param == nil {
			continue
		}

		grad := param.Grad

		// Initialize momentum and velocity if not present
		if _, exists := a.m[param]; !exists {
			a.m[param] = 0
			a.v[param] = 0
		}

		// Update biased first moment estimate
		a.m[param] = a.beta1*a.m[param] + (1-a.beta1)*grad

		// Update biased second moment estimate
		a.v[param] = a.beta2*a.v[param] + (1-a.beta2)*grad*grad

		// Compute bias-corrected first moment estimate
		mHat := a.m[param] / (1 - math.Pow(a.beta1, float64(a.t)))

		// Compute bias-corrected second moment estimate
		vHat := a.v[param] / (1 - math.Pow(a.beta2, float64(a.t)))

		// Update parameters
		param.Value -= a.lr * mHat / (math.Sqrt(vHat) + a.epsilon)
	}
}
