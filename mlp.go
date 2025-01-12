package main

type MLP struct {
	InputDim         int
	OutputDim        int
	Dims             []int
	TotalLayersCount int

	Layers []*Layer
}

func NewMLP(inputDims int, layers []*Layer) *MLP {
	mlp := new(MLP)
	mlp.InputDim = inputDims
	mlp.OutputDim = layers[len(layers)-1].OutDim
	mlp.Dims = make([]int, 0)
	mlp.Dims = append(mlp.Dims, inputDims)
	for _, layer := range layers {
		mlp.Dims = append(mlp.Dims, layer.OutDim)
	}
	mlp.TotalLayersCount = len(layers) + 1
	mlp.Layers = layers
	return mlp
}

func (m *MLP) Call(inputs []*Value) []*Value {
	t := inputs
	for _, layer := range m.Layers {
		t = layer.Call(t)
	}
	return t
}

func (m *MLP) Parameters() []*Value {
	params := make([]*Value, 0)
	for _, n := range m.Layers {
		params = append(params, n.Parameters()...)
	}
	return params
}
