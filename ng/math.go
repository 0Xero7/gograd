package ng

import "math/rand"

func Multinomial(probs []float64) int {
	cumsum := make([]float64, len(probs))
	cumsum[0] = probs[0]
	for i := 1; i < len(probs); i++ {
		cumsum[i] = probs[i] + cumsum[i-1]
	}

	val := rand.Float64()
	for i := range cumsum {
		if val <= cumsum[i] {
			return i
		}
	}

	return len(probs) - 1
}
