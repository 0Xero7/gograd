package optimizers

import (
	"fmt"
	"gograd/ng"
	"math"
)

type AdamTensor struct {
	lr           float64
	beta1, beta2 float64
	epsilon      float64
	m, v         [][]float64
	t            int

	effectiveBeta1, effectiveBeta2 float64
	params                         []*ng.Tensor
}

func NewAdamTensor(lr float64, params []*ng.Tensor) *AdamTensor {
	collect := make([]*ng.Tensor, 0)
	for _, w := range params {
		if w != nil && w.RequiresOptimization {
			collect = append(collect, w)
		}
	}

	m := make([][]float64, len(collect))
	v := make([][]float64, len(collect))
	for i, w := range collect {
		m[i] = make([]float64, w.Len())
		v[i] = make([]float64, w.Len())

		for index := range w.Len() {
			m[i][index] = 0
			v[i][index] = 0
		}
	}

	fmt.Println(len(m), len(v), len(params))

	return &AdamTensor{
		lr:      lr,
		beta1:   0.9,   // Exponential decay rate for first moment
		beta2:   0.999, // Exponential decay rate for second moment
		epsilon: 1e-8,  // Small constant to prevent division by zero
		m:       m,
		v:       v,
		t:       0,
		params:  collect,

		effectiveBeta1: 0.9,
		effectiveBeta2: 0.999,
	}
}

func (a *AdamTensor) Step() {
	// fmt.Println("UwU")
	a.t++
	a.effectiveBeta1 *= a.beta1
	a.effectiveBeta2 *= a.beta2

	mHatFactor := 1.0 / (1.0 - a.effectiveBeta1)
	vHatFactor := 1.0 / (1.0 - a.effectiveBeta2)
	// mHatD := a.beta1 * 0.9   // binPow(a.beta1, a.t)
	// vHatD := a.beta2 * 0.999 // binPow(a.beta2, a.t)

	for i, paramx := range a.params {
		for index := range a.params[i].Len() {
			am := a.m[i][index]
			av := a.v[i][index]
			pg := paramx.Grad[index]

			// Update biased first moment estimate
			a.m[i][index] = a.beta1*am + (1-a.beta1)*pg
			am = a.m[i][index]

			// Update biased second moment estimate
			a.v[i][index] = a.beta2*av + (1-a.beta2)*pg*pg
			av = a.v[i][index]

			// Compute bias-corrected first moment estimate
			mHat := am * mHatFactor

			// Compute bias-corrected second moment estimate
			vHat := av * vHatFactor

			// Update parameters
			paramx.Value[index] -= a.lr * mHat / (math.Sqrt(vHat) + a.epsilon)
		}
	}
}
