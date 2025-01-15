package initializers

import (
	"math"
	"math/rand"
)

type HeInitializer struct{}

func (h *HeInitializer) Sample(fanIn, fanOut int) float64 {
	stddev := math.Sqrt(2.0 / float64(fanIn))
	weight := rand.NormFloat64() * stddev
	return weight
}
