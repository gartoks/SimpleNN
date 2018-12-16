using System;
using System.Text;
using SimpleNN.Data;
using SimpleXML;

namespace SimpleNN.Layers {
    public class ConvolutionalLayer : Layer {
        private readonly ActivationFunctions.ActivationFunction Activation;

        public readonly int FilterSize;
        public readonly int Stride;
        public readonly int ZeroPadding;

        private readonly float[] weights;
        private readonly float[] biases;

        public ConvolutionalLayer(int filterCount, int filterSize, int stride, int zeroPadding, Layer previousLayer, ActivationFunctions.ActivationFunction activation)
            : base(CalculateWidth(filterSize, stride, zeroPadding, previousLayer), CalculateHeight(filterSize, stride, zeroPadding, previousLayer), filterCount, previousLayer) {

            Activation = activation;

            FilterSize = filterSize;
            Stride = stride;
            ZeroPadding = zeroPadding;

            this.weights = new float[filterCount * filterSize * filterSize * previousLayer.Depth];
            this.biases = new float[filterCount];
        }

        internal override NNDetailedFeedData FeedForward(NNFeedData input) {
            float[] rawOutput = new float[Neurons];

            for (int fIdx = 0; fIdx < FilterCount; fIdx++) {
                for (int yIdx = 0; yIdx < Height; yIdx++) {
                    for (int xIdx = 0; xIdx < Width; xIdx++) {
                        float sum = this.biases[fIdx];

                        for (int fzIdx = 0; fzIdx < PreviousLayer.Depth; fzIdx++) {
                            for (int fyIdx = 0; fyIdx < FilterSize; fyIdx++) {
                                for (int fxIdx = 0; fxIdx < FilterSize; fxIdx++) {
                                    float weight = this.weights[ToWeightIndex(fxIdx, fyIdx, fzIdx, fIdx)];

                                    sum += GetInputValue(input, xIdx, yIdx, fxIdx, fyIdx, fzIdx) * weight;
                                }
                            }
                        }

                        rawOutput[ConvertToNeuronIndex(xIdx, yIdx, fIdx)] = sum + this.biases[fIdx];
                    }
                }
            }

            // Apply activation function
            float[] output = new float[Neurons];
            for (int oIdx = 0; oIdx < Neurons; oIdx++)
                output[oIdx] = Activation.Function(rawOutput[oIdx], rawOutput);

            return new NNDetailedFeedData(this, output, rawOutput);
        }

        internal override void PropagateBackward(NNDetailedBackpropagationData backpropagationData) {
            float[] lastRawOutput = backpropagationData.FeedForwardData[this].RawOutputData;

            float[] dE_dz = new float[Neurons];
            float[] newWeights = new float[this.weights.Length];

            // For each filter
            for (int fIdx = 0; fIdx < FilterCount; fIdx++) {
                float dE_db = 0;
                float[] dE_dw = new float[FilterSize * FilterSize * PreviousLayer.Depth];

                for (int yIdx = 0; yIdx < Height; yIdx++) {
                    for (int xIdx = 0; xIdx < Width; xIdx++) {

                        int oIdx = ConvertToNeuronIndex(xIdx, yIdx, fIdx);

                        float do_dz = Activation.Gradient(lastRawOutput[oIdx], lastRawOutput);

                        float dE_do;
                        if (NextLayer == null) {
                            float o = backpropagationData.FeedForwardData[this].OutputData[oIdx];
                            dE_do = backpropagationData.ErrorGradient(o, oIdx);
                        } else {
                            dE_do = NextLayer.CalculateErrorByOutputGradient(backpropagationData, oIdx);
                        }

                        float dE_dz_tmp = dE_do * do_dz;
                        dE_dz[oIdx] = dE_dz_tmp;

                        for (int fzIdx = 0; fzIdx < PreviousLayer.Depth; fzIdx++) {
                            for (int fyIdx = 0; fyIdx < FilterSize; fyIdx++) {
                                for (int fxIdx = 0; fxIdx < FilterSize; fxIdx++) {
                                    float dz_dw = backpropagationData.FeedForwardData[PreviousLayer].OutputData[PreviousLayer.ConvertToNeuronIndex(xIdx * Stride + fxIdx, yIdx * Stride + fyIdx, fzIdx)];

                                    dE_dw[ToWeightIndex(fxIdx, fyIdx, fzIdx, 0)] += dE_dz_tmp * dz_dw;
                                }
                            }
                        }

                        dE_db += dE_do * do_dz;   // dz_dw = 1 for bias
                    }
                }

                for (int fzIdx = 0; fzIdx < PreviousLayer.Depth; fzIdx++) {
                    for (int fyIdx = 0; fyIdx < FilterSize; fyIdx++) {
                        for (int fxIdx = 0; fxIdx < FilterSize; fxIdx++) {

                            int weightIndex = ToWeightIndex(fxIdx, fyIdx, fzIdx, fIdx);
                            newWeights[weightIndex] = backpropagationData.CalculateNewWeight(this.weights[weightIndex], dE_dw[ToWeightIndex(fxIdx, fyIdx, fzIdx, 0)], this, weightIndex);
                        }
                    }
                }

                this.biases[fIdx] += backpropagationData.CalculateNewWeight(this.biases[fIdx], dE_db, this, fIdx);
            }

            backpropagationData.dE_dz[this] = dE_dz;
            backpropagationData.UpdatedWeights[this] = newWeights;
        }

        private float GetInputValue(NNFeedData input, int cx, int cy, int fx, int fy, int d) {
            int x = cx * Stride + fx - ZeroPadding;
            int y = cy * Stride + fy - ZeroPadding;

            if (x < 0 || x >= input.Width || y < 0 || y >= input.Height)
                return 0;

            return input[input.ToNeuronIndex(cx * Stride + fx, cy * Stride + fy, d)];
        }

        internal override float CalculateErrorByOutputGradient(NNDetailedBackpropagationData detailedBackpropagationData, int fromNeuronIndex) {
            float[] dE_dz = detailedBackpropagationData.dE_dz[this];

            var idcs = PreviousLayer.ConvertFromNeuronIndex(fromNeuronIndex);
            int x = idcs.x;
            int y = idcs.x;
            int z = idcs.z;

            float dE_do = 0;
            for (int zIdx = 0; zIdx < Depth; zIdx++) {
                for (int yIdx = 0; yIdx < Height; yIdx++) {
                    for (int xIdx = 0; xIdx < Width; xIdx++) {
                        int wx = xIdx - x;
                        int wy = yIdx - y;

                        if (wx < 0 || wx >= FilterSize || wy < 0 || wy >= FilterSize)
                            continue;

                        dE_do += this.weights[ToWeightIndex(wx, wy, z, zIdx)] * dE_dz[ConvertToNeuronIndex(xIdx, yIdx, zIdx)];
                    }
                }
            }

            return dE_do;
        }

        internal override void SetWeight(int weightIndex, float value) {
            this.weights[weightIndex] = value;
        }

        internal float GetWeight(int filterX, int filterY, int filterZ, int filter) {
            if (filterX < 0 || filterX >= FilterSize || filterY < 0 || filterY >= FilterSize || filterZ < 0 || filterZ >= PreviousLayer.Depth || filter < 0 || filter >= FilterCount)
                throw new IndexOutOfRangeException();

            return this.weights[ToWeightIndex(filterX, filterY, filterZ, filter)];
        }

        internal override int Weights => FilterCount * FilterSize * FilterSize * PreviousLayer.Depth;

        internal override void SetBias(int biasIndex, float value) {
            this.biases[biasIndex] = value;
        }

        internal override int Biases => FilterCount;

        internal override XMLElement ToXML() {
            XMLElement e = XMLElement.Create(GetType().Name);
            e.AddAttribute("FilterCount", $"{FilterCount}");
            e.AddAttribute("FilterSize", $"{FilterSize}");
            e.AddAttribute("Stride", $"{Stride}");
            e.AddAttribute("ZeroPadding", $"{ZeroPadding}");
            e.AddChild(Activation.ToXML());

            StringBuilder sb = new StringBuilder();

            XMLElement weightsXE = XMLElement.Create("Weights");
            e.AddChild(weightsXE);
            for (int i = 0; i < this.weights.Length; i++) {
                if (i != 0)
                    sb.Append(",");
                sb.Append(this.weights[i]);
            }
            weightsXE.Value = sb.ToString();

            sb.Clear();

            XMLElement biasesXE = XMLElement.Create("Biases");
            e.AddChild(biasesXE);
            for (int i = 0; i < this.biases.Length; i++) {
                if (i != 0)
                    sb.Append(",");
                sb.Append(this.biases[i]);
            }
            biasesXE.Value = sb.ToString();

            return e;
        }

        public int FilterCount => Depth;

        private int ToWeightIndex(int fx, int fy, int fz, int f) {
            return fx + FilterSize * (fy + FilterSize * (fz + PreviousLayer.Depth * f));
        }

        private (int fxIdx, int fyIdx, int fzIdx, int fIdx) FromWeightIndex(int nIdx) {
            int wh = FilterSize * FilterSize;
            int whd = wh * PreviousLayer.Depth;

            int fIdx = nIdx / whd;
            nIdx %= whd;

            int fzIdx = nIdx / wh;
            nIdx %= wh;

            int fyIdx = nIdx / FilterSize;
            int fxIdx = nIdx % FilterSize;

            return (fxIdx, fyIdx, fzIdx, fIdx);
        }

        private static int CalculateWidth(int filterSize, int stride, int zeroPadding, Layer previousLayer) {
            return (previousLayer.Width - filterSize + 2 * zeroPadding) / stride + 1;
        }

        private static int CalculateHeight(int filterSize, int stride, int zeroPadding, Layer previousLayer) {
            return (previousLayer.Height - filterSize + 2 * zeroPadding) / stride + 1;
        }
    }
}