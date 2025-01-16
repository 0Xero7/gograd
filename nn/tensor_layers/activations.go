package tensorlayers

type ActivationFunction int

const (
	relu ActivationFunction = iota
	tanh
	sigmoid
	softmax
)
