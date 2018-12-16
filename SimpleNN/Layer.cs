using System;
using SimpleNN.Data;
using SimpleNN.Layers;
using SimpleXML;

namespace SimpleNN {
    public abstract class Layer {

        internal static Layer FromXML(XMLElement root, Layer previousLayer) {

            float[] StringToArray(string s) {
                string[] valStrs = s.Split(',');
                float[] array = new float[valStrs.Length];
                for (int i = 0; i < valStrs.Length; i++) {
                    array[i] = float.Parse(valStrs[i]);
                }
                return array;
            }

            string type = root.Tag;
            if (type == typeof(InputLayer).Name) {
                int width = int.Parse(root.GetAttribute("Width").Value);
                int height = int.Parse(root.GetAttribute("Height").Value);
                int depth = int.Parse(root.GetAttribute("Depth").Value);

                return new InputLayer(width, height, depth);
            }

            if (type == typeof(MaxPoolingLayer).Name) {
                int filterSize = int.Parse(root.GetAttribute("FilterSize").Value);
                int stride = int.Parse(root.GetAttribute("Stride").Value);

                return new MaxPoolingLayer(filterSize, stride, previousLayer);
            }

            if (type == typeof(FullyConnectedLayer).Name) {
                int neurons = int.Parse(root.GetAttribute("Neurons").Value);
                ActivationFunctions.ActivationFunction activation = ActivationFunctions.FromXML(root.ChildWithTag("ActivationFunction"));

                FullyConnectedLayer layer = new FullyConnectedLayer(neurons, previousLayer, activation);

                float[] weights = StringToArray(root.ChildWithTag("Weights").Value);
                for (int i = 0; i < layer.Weights; i++) {
                    layer.SetWeight(i, weights[i]);
                }

                float[] biases = StringToArray(root.ChildWithTag("Biases").Value);
                for (int i = 0; i < layer.Biases; i++) {
                    layer.SetBias(i, biases[i]);
                }

                return layer;
            }

            if (type == typeof(ConvolutionalLayer).Name) {
                int filterCount = int.Parse(root.GetAttribute("FilterCount").Value);
                int filterSize = int.Parse(root.GetAttribute("FilterSize").Value);
                int stride = int.Parse(root.GetAttribute("Stride").Value);
                int zeroPadding = int.Parse(root.GetAttribute("ZeroPadding").Value);
                ActivationFunctions.ActivationFunction activation = ActivationFunctions.FromXML(root.ChildWithTag("ActivationFunction"));

                ConvolutionalLayer layer = new ConvolutionalLayer(filterCount, filterSize, stride, zeroPadding, previousLayer, activation);

                float[] weights = StringToArray(root.ChildWithTag("Weights").Value);
                for (int i = 0; i < layer.Weights; i++) {
                    layer.SetWeight(i, weights[i]);
                }

                float[] biases = StringToArray(root.ChildWithTag("Biases").Value);
                for (int i = 0; i < layer.Biases; i++) {
                    layer.SetBias(i, biases[i]);
                }

                return layer;
            }

            throw new NotImplementedException();    // Never reached
        }

        protected readonly Layer PreviousLayer;
        protected Layer NextLayer { get; private set; }

        public int Width { get; }
        public int Height { get; }
        public int Depth { get; }

        protected Layer(int width, int height, int depth) {
            Width = width;

            Height = height;
            Depth = depth;
        }

        protected Layer(int width, int height, int depth, Layer previousLayer) 
            : this(width, height, depth) {

            PreviousLayer = previousLayer;
            PreviousLayer.NextLayer = this;
        }

        internal abstract NNDetailedFeedData FeedForward(NNFeedData input);

        internal abstract void PropagateBackward(NNDetailedBackpropagationData backpropagationData);

        public void InitializeWeights(Func<float> generator) {
            for (int i = 0; i < Weights; i++) {
                SetWeight(i, generator());
            }
        }

        internal int ConvertToNeuronIndex(int x, int y, int z) => x + Width * (y + Height * z);

        internal (int x, int y, int z) ConvertFromNeuronIndex(int nIdx) {
            int wh = Width * Height;
            int z = nIdx / wh;
            nIdx %= wh;
            int y = nIdx / Width;
            int x = nIdx % Width;

            return (x, y, z);
        }

        internal abstract float CalculateErrorByOutputGradient(NNDetailedBackpropagationData detailedBackpropagationData, int fromNeuronIndex);

        internal abstract void SetWeight(int weightIndex, float value);

        internal abstract int Weights { get; }

        internal abstract void SetBias(int biasIndex, float value);

        internal abstract int Biases { get; }

        internal int Neurons => Width * Height * Depth;

        internal abstract XMLElement ToXML();
    }
}