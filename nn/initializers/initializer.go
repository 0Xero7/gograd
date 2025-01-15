package initializers

type Initializer interface {
	Sample(fanIn, fanOut int) float64
}
