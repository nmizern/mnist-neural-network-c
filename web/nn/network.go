package nn

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

type Layer struct {
	Rows    uint64
	Cols    uint64
	Weights []float32
}

type Network struct {
	NumLayers  uint64
	LayerSizes []uint64
	Layers     []Layer
}

func LoadModel(path string) (*Network, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open model: %w", err)
	}
	defer f.Close()

	var net Network

	if err := binary.Read(f, binary.LittleEndian, &net.NumLayers); err != nil {
		return nil, fmt.Errorf("failed to read num_layers: %w", err)
	}

	net.LayerSizes = make([]uint64, net.NumLayers+1)
	if err := binary.Read(f, binary.LittleEndian, &net.LayerSizes); err != nil {
		return nil, fmt.Errorf("failed to read layer_sizes: %w", err)
	}

	net.Layers = make([]Layer, net.NumLayers)
	for i := uint64(0); i < net.NumLayers; i++ {
		var rows, cols uint64
		if err := binary.Read(f, binary.LittleEndian, &rows); err != nil {
			return nil, fmt.Errorf("failed to read layer %d dimensions: %w", i, err)
		}
		if err := binary.Read(f, binary.LittleEndian, &cols); err != nil {
			return nil, fmt.Errorf("failed to read layer %d dimensions: %w", i, err)
		}

		weights := make([]float32, rows*cols)
		if err := binary.Read(f, binary.LittleEndian, &weights); err != nil {
			return nil, fmt.Errorf("failed to read layer %d weights: %w", i, err)
		}

		net.Layers[i] = Layer{Rows: rows, Cols: cols, Weights: weights}
	}

	fmt.Printf("Model loaded: %d layers %v\n", net.NumLayers, net.LayerSizes)
	return &net, nil
}

func sigmoid(x float32) float32 {
	if x < -500 {
		x = -500
	}
	if x > 500 {
		x = 500
	}
	return float32(1.0 / (1.0 + math.Exp(float64(-x))))
}

func (net *Network) Predict(input []float32) (int, []float32) {
	current := input

	for _, layer := range net.Layers {
		biasInput := make([]float32, len(current)+1)
		copy(biasInput, current)
		biasInput[len(current)] = 1.0

		output := make([]float32, layer.Rows)
		for i := uint64(0); i < layer.Rows; i++ {
			var sum float32
			for j := uint64(0); j < layer.Cols; j++ {
				sum += layer.Weights[i*layer.Cols+j] * biasInput[j]
			}
			output[i] = sigmoid(sum)
		}

		current = output
	}

	maxIdx := 0
	for i := 1; i < len(current); i++ {
		if current[i] > current[maxIdx] {
			maxIdx = i
		}
	}

	return maxIdx, current
}
