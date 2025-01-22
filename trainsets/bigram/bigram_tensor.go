package bigram

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
)

var vocab []string
var vocabMap map[string]int
var reverseVocabMap map[int]string
var vocabSize int

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

func LoadDataset() {
	data, err := os.ReadFile("makemore/names.txt")
	if err != nil {
		panic(err)
	}

	_vocabMap := make(map[string]bool, 0)
	vocabMap = make(map[string]int, 0)
	reverseVocabMap = make(map[int]string, 0)
	datasetNames = make([]string, 0)

	names := strings.Split(string(data), "\n")
	for i := range names {
		names[i] = "#" + strings.TrimSpace(names[i]) + "."
		datasetNames = append(datasetNames, names[i])

		for _, r := range names[i] {
			_vocabMap[string(r)] = true
		}
	}

	for k := range _vocabMap {
		vocab = append(vocab, k)
		vocabMap[k] = len(vocabMap)
		reverseVocabMap[vocabMap[k]] = k
	}
}

func TrainBigram(iterations, batchSize int, learningRate float64) *nn.MLPTensor {
	vocabSize = len(vocab)

	mlp := nn.NewMLPTensor(vocabSize, []nn.TensorLayer{
		tensorlayers.Linear(vocabSize, vocabSize, &initializers.XavierInitializer{}),
		// tensorlayers.Tanh(512),

		// tensorlayers.Linear(512, 128, &initializers.XavierInitializer{}),
		// tensorlayers.Tanh(128),

		// tensorlayers.Linear(128, len(vocab), &initializers.XavierInitializer{}),
	})

	optimizer := optimizers.NewAdamTensor(learningRate, mlp.Parameters())

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
			for j := 0; j+1 < len(str); j++ {
				a, r := runeToToken(string(str[j])), runeToToken(string(str[j+1]))
				xsi = append(xsi, ng.OneHot(a, vocabSize))
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
		runtime.GC()

		// ng.TTensorPool.Reset()
		perf.PrintMemStats()

		fmt.Printf("Epoch %d complete. Loss = %.4f\n", epoch+1, lossValue)
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

func Predict(mlp *nn.MLPTensor, s string) {
	input := s
	// fmt.Print(s)

	for {
		xs := ng.Tensor2D([][]float64{
			ng.OneHot(runeToToken(input), vocabSize),
		})

		logits := mlp.Call(xs).SoftMax(1)
		// fmt.Println(logits)
		// break

		sel := ng.Multinomial(logits.Value)
		char := tokenToRune(sel)
		if char == "." {
			fmt.Println()
			return
		}

		input = char
		fmt.Print(char)
	}
}
