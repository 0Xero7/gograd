package initializers

import (
	"math"
	"math/rand"
)

type XavierInitializer struct{}

func (h *XavierInitializer) Sample(fanIn, fanOut int) float64 {
	stddev := math.Sqrt(2.0 / float64(fanIn+fanOut))
	return rand.NormFloat64() * stddev
}
