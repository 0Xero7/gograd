package optimizers

import (
	"gograd/ng"
	"math"
)

type AdamTensor struct {
	lr           float64
	beta1, beta2 float64
	epsilon      float64
	m, v         map[int][]float64
	t            int

	effectiveBeta1, effectiveBeta2 float64
}

func NewAdamTensor(lr float64) *AdamTensor {
	return &AdamTensor{
		lr:      lr,
		beta1:   0.9,   // Exponential decay rate for first moment
		beta2:   0.999, // Exponential decay rate for second moment
		epsilon: 1e-8,  // Small constant to prevent division by zero
		m:       make(map[int][]float64),
		v:       make(map[int][]float64),
		t:       0,

		effectiveBeta1: 0.9,
		effectiveBeta2: 0.999,
	}
}

func binPow(inp float64, power int) float64 {
	if power == 0 {
		return 1
	}
	if power == 1 {
		return inp
	}

	m := binPow(inp, power/2)
	m = m * m
	if power%2 == 1 {
		m = m * inp
	}
	return m
}

func (a *AdamTensor) Step(params []*ng.Tensor) {
	// fmt.Println("UwU")
	a.t++
	a.effectiveBeta1 *= a.beta1
	a.effectiveBeta2 *= a.beta2

	mHatFactor := 1.0 / (1.0 - a.effectiveBeta1)
	vHatFactor := 1.0 / (1.0 - a.effectiveBeta2)
	// mHatD := a.beta1 * 0.9   // binPow(a.beta1, a.t)
	// vHatD := a.beta2 * 0.999 // binPow(a.beta2, a.t)

	for _, paramx := range params {
		if paramx == nil || !paramx.RequiresOptimization {
			continue
		}
		param := paramx.Id

		// grad := param.Grad

		// Initialize momentum and velocity if not present
		if _, exists := a.m[param]; !exists {
			a.m[param] = make([]float64, paramx.Len())
			a.v[param] = make([]float64, paramx.Len())
		}

		for index := range paramx.Len() {
			am := a.m[param][index]
			av := a.v[param][index]
			pg := paramx.Grad[index]

			// Update biased first moment estimate
			a.m[param][index] = a.beta1*am + (1-a.beta1)*pg
			am = a.m[param][index]

			// Update biased second moment estimate
			a.v[param][index] = a.beta2*av + (1-a.beta2)*pg*pg
			av = a.v[param][index]

			// Compute bias-corrected first moment estimate
			mHat := am * mHatFactor

			// Compute bias-corrected second moment estimate
			vHat := av * vHatFactor

			// Update parameters
			paramx.Value[index] -= a.lr * mHat / (math.Sqrt(vHat) + a.epsilon)
		}
	}
}
