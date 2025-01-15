package initializers

import (
	"math/rand"
)

type SimpleInitializer struct{}

func (h *SimpleInitializer) Sample(fanIn, fanOut int) float64 {
	return (rand.Float64() - 0.5) * 2
}
