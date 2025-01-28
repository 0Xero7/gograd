package shakespeare

import (
	"fmt"
	"gograd/ng"
	"gograd/nn"
	"gograd/nn/initializers"
	lossfunctions "gograd/nn/loss_functions"
	tensorlayers "gograd/nn/tensor_layers"
	"gograd/optimizers"
	"gograd/perf"
	"math/rand"
	"os"
	"runtime"
	"slices"
	"strings"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

var vocab []string
var vocabMap map[string]int
var reverseVocabMap map[int]string
var vocabSize int

var embeddingSize int = 16
var embeddings [][]float64

func runeToToken(r string) int {
	return vocabMap[r]
}

func tokenToRune(token int) string {
	return reverseVocabMap[token]
}

func stringToTokens(s string) []float64 {
	tokens := make([]float64, len(s))
	for i, r := range s {
		tokens[i] = float64(runeToToken(string(r)))
	}
	return tokens
}

func tokensToString(t []float64) string {
	s := ""
	for _, i := range t {
		s += string(tokenToRune(int(i)))
	}
	return s
}

var datasetNames []string

func LoadDataset(gram int) {
	data, err := os.ReadFile("makemore/tinyshakespeare.txt")
	if err != nil {
		panic(err)
	}

	_vocabMap := make(map[string]bool, 0)
	vocabMap = make(map[string]int, 0)
	reverseVocabMap = make(map[int]string, 0)
	datasetNames = make([]string, 0)

	prefix := ""
	for range gram {
		prefix = prefix + "#"
	}

	words := make(map[string]bool, 0)

	names := strings.Split(string(data), "\n")
	for i := range names {
		temp := strings.Split(names[i], " ")
		for j := range temp {
			words[temp[j]] = true
		}

		names[i] = prefix + strings.TrimSpace(names[i]) + "."
		datasetNames = append(datasetNames, names[i])

		for _, r := range names[i] {
			_vocabMap[string(r)] = true
		}
	}

	fmt.Println("Unique words:", len(words))

	for k := range _vocabMap {
		vocab = append(vocab, k)
		vocabMap[k] = len(vocabMap)
		reverseVocabMap[vocabMap[k]] = k
		embeddings = append(embeddings, ng.TensorRand(embeddingSize).Value)
	}
}

func TrainShakespeare(gram, iterations, batchSize int, learningRate float64) *nn.MLPTensor {
	vocabSize = len(vocab)

	mlp := nn.NewMLPTensor(gram*embeddingSize, []nn.TensorLayer{
		tensorlayers.Linear(gram*embeddingSize, 256, &initializers.HeInitializer{}),
		tensorlayers.ReLu(256),

		tensorlayers.Linear(256, 256, &initializers.HeInitializer{}),
		tensorlayers.ReLu(256),

		tensorlayers.Linear(256, 256, &initializers.HeInitializer{}),
		tensorlayers.ReLu(256),

		tensorlayers.Linear(256, vocabSize, &initializers.XavierInitializer{}),
	})

	fmt.Println(mlp.ParameterCount())

	optimizer := optimizers.NewAdamTensor(learningRate, mlp.Parameters())

	losses := make([]float64, 0)

	for epoch := range iterations {
		batchIndices := make([]int, 0)
		for len(batchIndices) < batchSize {
			nextIndex := rand.Intn(len(datasetNames))
			if slices.Contains(batchIndices, nextIndex) {
				continue
			}

			batchIndices = append(batchIndices, nextIndex)
		}

		xsi := make([][]float64, 0)
		ysi := make([][]float64, 0)
		for _, i := range batchIndices {
			str := datasetNames[i]
			for j := 0; j+gram < len(str); j++ {
				for delta := range gram {
					a := runeToToken(string(str[j+delta]))
					xsi = append(xsi, embeddings[a]) //ng.OneHot(a, vocabSize))
				}
				r := runeToToken(string(str[j+gram]))
				ysi = append(ysi, ng.OneHot(r, vocabSize))

				// fmt.Println(a, b, ">", r)
			}
		}

		// fmt.Println(xsi)

		xs := ng.Tensor2D(xsi)
		ys := ng.Tensor2D(ysi)

		logits := mlp.Call(xs)
		loss := lossfunctions.TensorCrossEntropyProbDist(logits, ys)
		loss.Backward()
		optimizer.Step()

		lossValue := loss.Value[0]
		loss = nil
		losses = append(losses, lossValue)

		// ng.TTensorPool.Reset()

		ng.ResetPoolIndex()
		if epoch%1 == 0 {
			runtime.GC()
			perf.PrintMemStats()
			fmt.Printf("Epoch %d complete. Loss = %.4f\n", epoch+1, lossValue)
		}
	}

	// Create a new plot
	p := plot.New()
	p.Title.Text = "Training Loss"
	p.X.Label.Text = "Iteration"
	p.Y.Label.Text = "Loss"

	// Prepare data points
	pts := make(plotter.XYs, len(losses))
	for i, loss := range losses {
		pts[i].X = float64(i)
		pts[i].Y = loss
	}

	// Create a line plotter
	line, err := plotter.NewLine(pts)
	if err != nil {
		panic(err)
	}
	p.Add(line)

	// Save the plot to a PNG file
	if err := p.Save(8*vg.Inch, 8*vg.Inch, "loss.png"); err != nil {
		panic(err)
	}

	return mlp
}

func argmax(vals []float64) int {
	best := -1
	bestVal := -100000000000.0

	for i := range vals {
		if vals[i] > bestVal {
			bestVal = vals[i]
			best = i
		}
	}

	return best
}

func Predict(mlp *nn.MLPTensor, gram int) {
	input := ""
	for range gram {
		input += "#"
	}
	// fmt.Print(s)

	for {
		xsi := make([][]float64, 0)
		for i := range gram {
			xsi = append(xsi, embeddings[runeToToken(string(input[i]))])
		}

		xs := ng.Tensor2D(xsi)

		logits := mlp.Call(xs).SoftMax(1)
		// fmt.Println(logits)
		// break

		sel := ng.Multinomial(logits.Value)
		char := tokenToRune(sel)
		if char == "." {
			fmt.Println()
			return
		}

		ng.ResetPoolIndex()
		input = string(input[1:]) + char
		fmt.Print(char)
	}
}
