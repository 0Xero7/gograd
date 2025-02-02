package tracers

import (
	"bytes"
	"context"
	"fmt"
	"gograd/ng"
	"log"
	"os"

	"github.com/goccy/go-graphviz"
	"github.com/goccy/go-graphviz/cgraph"
)

func getTensorNodeID(v *ng.Tensor) string {
	return fmt.Sprintf("%s", &v.Id)
}

func TraceTensor(root *ng.Tensor) {
	TraceTensor2(root, 0)
}

func TraceTensor2(root *ng.Tensor, idx int) {
	ctx := context.Background()
	g, err := graphviz.New(ctx)
	if err != nil {
		panic(err)
	}
	graph, err := g.Graph(graphviz.WithDirectedType(graphviz.Directed))
	if err != nil {
		panic(err)
	}

	// Set graph direction left to right
	// graph.SetRankDir("LR")

	defer func() {
		if err := graph.Close(); err != nil {
			panic(err)
		}
		g.Close()
	}()

	nodes := make(map[*ng.Tensor]*cgraph.Node)
	var buildGraph func(v *ng.Tensor) *cgraph.Node
	buildGraph = func(v *ng.Tensor) *cgraph.Node {
		if node, exists := nodes[v]; exists {
			return node
		}

		// Create value node with box shape
		node, err := graph.CreateNodeByName(getTensorNodeID(v))
		limit := min(len(v.Value), 10)
		node.SetLabel(fmt.Sprintf("%d\ndata=%.4f\ngrad=%.8f", v.Id, v.Value[0:limit], v.Grad[0:limit]))
		// node.SetLabel(fmt.Sprintf("%d", v.Id))
		node.SetShape("box")
		node.SetStyle("filled")
		node.SetFillColor("lightblue")
		node.SetPenWidth(2.0)
		if err != nil {
			panic(err)
		}
		nodes[v] = node

		if v.Children.Len() > 0 && v.Op != "X" {
			// Create operation node with circle shape
			opNode, err := graph.CreateNodeByName(fmt.Sprintf("op_%s", getTensorNodeID(v)))
			if err != nil {
				panic(err)
			}
			opNode.SetLabel(string(v.Op))
			opNode.SetShape("circle")
			opNode.SetStyle("filled")
			opNode.SetFillColor("lightgreen")
			opNode.SetPenWidth(2.0)

			// Connect operation node to result with visible edge
			edge, err := graph.CreateEdgeByName("", opNode, node)
			edge.SetStyle(cgraph.BoldEdgeStyle)
			if err != nil {
				panic(err)
			}
			edge.SetPenWidth(1.5)

			// Connect children to operation node with visible edges
			for i := 0; i < v.Children.Len(); i++ {
				childNode := buildGraph(v.Children.At(i))
				edge, err := graph.CreateEdgeByName("", childNode, opNode)
				edge.SetStyle(cgraph.BoldEdgeStyle)
				if err != nil {
					panic(err)
				}
				edge.SetPenWidth(1.5)
			}
		}
		return node
	}
	buildGraph(root)

	var buf bytes.Buffer
	if err := g.Render(ctx, graph, "dot", &buf); err != nil {
		log.Fatal(err)
	}

	fmt.Println("[DEBUG]: Writing to ", fmt.Sprint("output/graph_", idx, ".dot"))
	err = os.WriteFile(fmt.Sprint("output/graph_", idx, ".dot"), buf.Bytes(), os.ModePerm)
	if err != nil {
		panic(err)
	}
}
