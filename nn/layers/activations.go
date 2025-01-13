package layers

type ActivationFunction int

const (
	relu ActivationFunction = iota
	tanh
	sigmoid
	// SoftMax
)
