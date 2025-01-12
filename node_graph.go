package main

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"os"

	"github.com/goccy/go-graphviz"
	"github.com/goccy/go-graphviz/cgraph"
)

func getNodeID(v *Value) string {
	return fmt.Sprintf("%p", v)
}

func trace(root *Value) {
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
	graph.SetRankDir("LR")

	defer func() {
		if err := graph.Close(); err != nil {
			panic(err)
		}
		g.Close()
	}()

	nodes := make(map[*Value]*cgraph.Node)
	var buildGraph func(v *Value) *cgraph.Node
	buildGraph = func(v *Value) *cgraph.Node {
		if node, exists := nodes[v]; exists {
			return node
		}

		// Create value node with box shape
		node, err := graph.CreateNodeByName(getNodeID(v))
		node.SetLabel(fmt.Sprintf("%s | data=%.4f | grad=%.4f", v.Label, v.Value, v.Grad))
		node.SetShape("box")
		node.SetStyle("filled")
		node.SetFillColor("lightblue")
		node.SetPenWidth(2.0)
		if err != nil {
			panic(err)
		}
		nodes[v] = node

		if len(v.Children) > 0 && v.Op != "X" {
			// Create operation node with circle shape
			opNode, err := graph.CreateNodeByName(fmt.Sprintf("op_%s", getNodeID(v)))
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
			for _, child := range v.Children {
				childNode := buildGraph(child)
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

	os.WriteFile("output/graph.dot", buf.Bytes(), os.ModePerm)
}
