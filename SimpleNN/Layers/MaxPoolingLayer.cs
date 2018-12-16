using System;
using SimpleNN.Data;
using SimpleXML;

namespace SimpleNN.Layers {
    public class MaxPoolingLayer : Layer {
        private int FilterSize { get; }
        private int Stride { get; }

        public MaxPoolingLayer(int filterSize, int stride, Layer previousLayer)
            : base(CalculateWidth(filterSize, stride, previousLayer), CalculateHeight(filterSize, stride, previousLayer), previousLayer.Depth, previousLayer) {

            this.FilterSize = filterSize;
            this.Stride = stride;
        }

        internal override NNDetailedFeedData FeedForward(NNFeedData input) {
            float[] output = new float[Neurons];
            int[] maxIndices = new int[Neurons];

            for (int zIdx = 0; zIdx < PreviousLayer.Depth; zIdx++) {
                for (int yiIdx = 0, yoIdx = 0; yoIdx < Height; yiIdx += this.Stride, yoIdx++) {
                    for (int xiIdx = 0, xoIdx = 0; xoIdx < Width; xiIdx += this.Stride, xoIdx++) {
                        float max = float.MinValue;
                        int maxIdx = -1;

                        for (int fyIdx = 0; fyIdx < this.FilterSize; fyIdx++) {
                            for (int fxIdx = 0; fxIdx < this.FilterSize; fxIdx++) {
                                int idx = PreviousLayer.ConvertToNeuronIndex(xiIdx + fxIdx, yiIdx + fyIdx, zIdx);

                                if (input[idx] > max) {
                                    max = input[idx];
                                    maxIdx = idx;
                                }
                            }
                        }

                        int i = ConvertToNeuronIndex(xoIdx, yoIdx, zIdx);
                        output[i] = max;
                        maxIndices[i] = maxIdx;
                    }
                }
            }

            NNDetailedFeedData feedData = new NNDetailedFeedData(this, output, output);
            feedData.CustomData[nameof(maxIndices)] = maxIndices;

            return feedData;
        }

        internal override void PropagateBackward(NNDetailedBackpropagationData backpropagationData) {
            float[] dE_dz = new float[Neurons];

            for (int oIdx = 0; oIdx < Neurons; oIdx++) {
                float dE_do;
                if (NextLayer == null) {
                    // Output of this neuron
                    float o = backpropagationData.FeedForwardData[this].OutputData[oIdx];

                    // Gradient of the error with respect to the neurons's output on the last layer is the gradient of the error function with the neuron's output
                    dE_do = backpropagationData.ErrorGradient(o, oIdx);
                } else {
                    // Gradient of error with respect to the neuron's output on any other layer is the sum
                    // of each weight connecting this neuron to every neuron in the next layer
                    // multiplied by that neuron's gradient of the error with respect to that neuron's raw output
                    dE_do = NextLayer.CalculateErrorByOutputGradient(backpropagationData, oIdx);
                }

                // Gradient of the error with respect to the raw output is the gradient of the error with respect to the output
                // multiplied by the gradient of the output with respect to the raw output.
                // Gradient of the output with respect to the neuron raw output (do/dz) is just one (as the activation function is f(x) = x and thus df(x)/dx = 1
                dE_dz[oIdx] = dE_do;
            }

            backpropagationData.dE_dz[this] = dE_dz;
            backpropagationData.UpdatedWeights[this] = new float[0];
        }


        internal override float CalculateErrorByOutputGradient(NNDetailedBackpropagationData detailedBackpropagationData, int fromNeuronIndex) {
            float[] dE_dz = detailedBackpropagationData.dE_dz[this];
            int[] maxIndices = (int[])detailedBackpropagationData.FeedForwardData[this].CustomData["maxIndices"];

            for (int i = 0; i < maxIndices.Length; i++) {   // maxIndices.Length == this.Neurons
                if (maxIndices[i] == fromNeuronIndex)
                    return dE_dz[i];
            }

            return 0;
        }

        internal override void SetWeight(int weightIndex, float value) {
            // Do nothing
        }

        private float GetWeight(int[] maxIndices, int fromNeuronIndex, int toNeuronIndex) {
            return maxIndices[toNeuronIndex] == fromNeuronIndex ? 1 : 0;
        }

        private static int CalculateWidth(int size, int stride, Layer previousLayer) {
            float wf = (previousLayer.Width - size) / (float)stride + 1;
            int w = (previousLayer.Width - size) / stride + 1;
            
            if (Math.Abs(w - wf) > 0.01)
                throw new ArgumentException("Filter size and stride invalid for previous' layer width.");

            return w;
        }

        private static int CalculateHeight(int size, int stride, Layer previousLayer) {
            float hf = (previousLayer.Height - size) / (float)stride + 1;
            int h = (previousLayer.Height - size) / stride + 1;

            if (Math.Abs(h - hf) > 0.01)
                throw new ArgumentException("Filter size and stride invalid for previous' layer height.");

            return h;
        }

        internal override int Weights => Neurons * PreviousLayer.Neurons;

        internal override void SetBias(int biasIndex, float value) {
            // Do nothing
        }

        internal override int Biases => 0;

        internal override XMLElement ToXML() {
            XMLElement e = XMLElement.Create(GetType().Name);
            e.AddAttribute("FilterSize", $"{FilterSize}");
            e.AddAttribute("Stride", $"{Stride}");
            return e;
        }
    }
}