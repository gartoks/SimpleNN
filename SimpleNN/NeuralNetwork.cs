using System;
using System.Collections.Generic;
using System.Linq;
using SimpleNN.Data;
using SimpleNN.Layers;
using SimpleXML;

namespace SimpleNN {
    public sealed class NeuralNetwork {
        public static NeuralNetwork FromXML(string xml) {
            XMLElement root = XMLReader.Parse(xml);

            int layerCount = int.Parse(root.GetAttribute("Layers").Value);

            XMLElement[] layerXEs = new XMLElement[layerCount];

            Layer[] layers = new Layer[layerCount];
            foreach (XMLElement layerXE in root.Children) {
                int layerIndex = int.Parse(layerXE.GetAttribute("Index").Value);

                layerXEs[layerIndex] = layerXE;
            }

            for (int i = 0; i < layerXEs.Length; i++) {
                Layer previousLayer = i == 0 ? null : layers[i - 1];

                Layer layer = Layer.FromXML(layerXEs[i], previousLayer);
                layers[i] = layer;
            }

            return new NeuralNetwork(layers);
        }

        private readonly Layer[] layers;

        public NeuralNetwork(params Layer[] layers)
            : this((IEnumerable<Layer>)layers) { }

        public NeuralNetwork(IEnumerable<Layer> layers) {
            if (!(layers.First() is InputLayer))
                throw new ArgumentException("The first layer of a neural network must be an input layer.");

            this.layers = new Layer[layers.Count()];
            int i = 0;
            foreach (Layer layer in layers) {
                this.layers[i++] = layer;
            }
        }

        public NNFeedData FeedForward(NNFeedData input) {
            NNFeedData data = input;
            for (int i = 0; i < this.layers.Length; i++) {
                Layer layer = this.layers[i];

                NNDetailedFeedData feedData = layer.FeedForward(data);
                data = feedData.OutputData;
            }

            return data;
        }

        public void PropagateBackward(NNBackpropagationData backpropagationData) {
            NNTrainingData trainingData = backpropagationData.TrainingData;

            // Validate training data dimensions
            if (trainingData.InputDataWidth != InputLayer.Width || trainingData.InputDataHeight != InputLayer.Height || trainingData.InputDataDepth != InputLayer.Depth)
                throw new ArgumentException();

            if (trainingData.TargetDataWidth != OutputLayer.Width || trainingData.TargetDataHeight != OutputLayer.Height || trainingData.TargetDataDepth != OutputLayer.Depth)
                throw new ArgumentException();

            for (int trainingDataIndex = 0; trainingDataIndex < trainingData.DataSize; trainingDataIndex++) {
                backpropagationData.BatchTrainingStartingCallback?.Invoke(trainingDataIndex, trainingData.DataSize);
                
                // Initialize backpropagation run
                NNDetailedBackpropagationData detailedBackpropagationData = new NNDetailedBackpropagationData(backpropagationData, trainingDataIndex);

                // Feed forward through the network and gather neccessary data
                NNFeedData feedForwardInputData = trainingData.GetInputData(trainingDataIndex);
                for (int i = 0; i < this.layers.Length; i++) {
                    Layer layer = this.layers[i];

                    NNDetailedFeedData feedData = layer.FeedForward(feedForwardInputData);
                    detailedBackpropagationData.FeedForwardData[layer] = feedData;

                    feedForwardInputData = feedData.OutputData;
                }

                // Propagate error backwards through the network
                for (int i = this.layers.Length - 1; i >= 0; i--) {
                    Layer layer = this.layers[i];

                    layer.PropagateBackward(detailedBackpropagationData);
                }

                // Update weights for each layer
                foreach (KeyValuePair<Layer, float[]> updatedWeight in detailedBackpropagationData.UpdatedWeights) {
                    Layer layer = updatedWeight.Key;
                    float[] weights = updatedWeight.Value;

                    for (int i = 0; i < weights.Length; i++)
                        layer.SetWeight(i, weights[i]);
                }

                backpropagationData.BatchTrainingFinishedCallback?.Invoke(trainingDataIndex, trainingData.DataSize);
            }
        }

        public Layer InputLayer => this.layers[0];

        public Layer OutputLayer => this.layers[this.layers.Length - 1];

        public string ToXML() {
            XMLElement rootXE = XMLElement.Create(GetType().Name);
            rootXE.AddAttribute("Layers", this.layers.Length.ToString());

            for (int i = 0; i < this.layers.Length; i++) {
                XMLElement layerXE = this.layers[i].ToXML();
                layerXE.AddAttribute("Index", $"{i}");

                rootXE. AddChild(layerXE);
            }

            return XMLWriter.Parse(rootXE);
        }
    }
}