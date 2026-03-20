package main

import (
	"flag"
	"log"
	"net/http"
	"nn-web/nn"

	"github.com/gin-gonic/gin"
)

type PredictRequest struct {
	Pixels []float32 `json:"pixels" binding:"required"`
}

type PredictResponse struct {
	Digit      int       `json:"digit"`
	Confidence []float32 `json:"confidence"`
}

func main() {
	modelPath := flag.String("model", "../data/mnist_model.bin", "path to trained model")
	addr := flag.String("addr", ":4343", "listen address")
	flag.Parse()

	model, err := nn.LoadModel(*modelPath)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()

	r.LoadHTMLGlob("templates/*")
	r.Static("/static", "./static")

	r.GET("/", func(c *gin.Context) {
		c.HTML(http.StatusOK, "index.html", nil)
	})

	r.POST("/api/predict", func(c *gin.Context) {
		var req PredictRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request"})
			return
		}

		if len(req.Pixels) != 784 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "expected 784 pixels"})
			return
		}

		digit, confidence := model.Predict(req.Pixels)
		c.JSON(http.StatusOK, PredictResponse{
			Digit:      digit,
			Confidence: confidence,
		})
	})

	log.Printf("Server starting on %s", *addr)
	if err := r.Run(*addr); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
