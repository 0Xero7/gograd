package main

type MLP struct {
	InputDim         int
	OutputDim        int
	Dims             []int
	TotalLayersCount int

	Layers []*Layer
}

func NewMLP(inputDims int, hiddenDims []int, outputDims int) *MLP {
	mlp := new(MLP)
	mlp.InputDim = inputDims
	mlp.OutputDim = outputDims
	mlp.Dims = make([]int, 0)
	mlp.Dims = append(mlp.Dims, inputDims)
	mlp.Dims = append(mlp.Dims, hiddenDims...)
	mlp.Dims = append(mlp.Dims, outputDims)
	mlp.TotalLayersCount = len(hiddenDims) + 2

	for i := range len(hiddenDims) + 1 {
		mlp.Layers = append(mlp.Layers, NewLayer(mlp.Dims[i], mlp.Dims[i+1]))
	}
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
