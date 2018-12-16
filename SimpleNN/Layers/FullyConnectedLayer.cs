using System.Text;
using SimpleNN.Data;
using SimpleXML;

namespace SimpleNN.Layers {
    public class FullyConnectedLayer : Layer {

        private readonly ActivationFunctions.ActivationFunction Activation;

        private readonly float[] weights;
        private readonly float[] biases;

        public FullyConnectedLayer(int neurons, Layer previousLayer, ActivationFunctions.ActivationFunction activation)
            : base(neurons, 1, 1, previousLayer) {

            Activation = activation;

            this.weights = new float[Neurons * PreviousLayer.Neurons];
            this.biases = new float[Neurons];
        }

        internal override NNDetailedFeedData FeedForward(NNFeedData input) {
            float[] rawOutput = new float[Neurons];

            for (int oIdx = 0; oIdx < Neurons; oIdx++) {
                // Add bias
                float rawOut = this.biases[oIdx];

                // Add up weights times previous output
                for (int iIdx = 0; iIdx < input.Size; iIdx++)
                    rawOut += this.weights[ToWeightIndex(iIdx, oIdx)] * input[iIdx];

                rawOutput[oIdx] = rawOut;
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
            float[] newWeights = new float[Neurons * PreviousLayer.Neurons];

            for (int oIdx = 0; oIdx < Neurons; oIdx++) {
                // The gradient of the error with respect to a specific weight is the
                // gradient of the error with respect to the output multiplied by the
                // gradient of the output with respect to the raw output multiplied by the
                // gradient of the raw output with respect to the specific weight
                // dE/dw_ij = dE/do_j * do_j/dz_j * dz_j/dw_ij
                
                // Calculate neurons specific values

                // Gradient of the output with respect to the neuron raw output is the gradient of the activation function with the raw output
                float do_dz = Activation.Gradient(lastRawOutput[oIdx], lastRawOutput);

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
                // multiplied by the gradient of the output with respect to the raw output
                dE_dz[oIdx] = dE_do * do_dz;

                // Calculate weight specific values
                for (int iIdx = 0; iIdx < PreviousLayer.Neurons; iIdx++) {
                    // Gradient of the raw output with respect to the specific weight  is the previous layer's output of the connected neuron
                    float dz_dw = backpropagationData.FeedForwardData[PreviousLayer].OutputData[iIdx];

                    float dE_dw = dE_do * do_dz * dz_dw;

                    // The change for the weight is the error gradient with respect to the weight multiplied by the learning rate
                    int weightIndex = ToWeightIndex(iIdx, oIdx);
                    newWeights[weightIndex] = backpropagationData.CalculateNewWeight(this.weights[weightIndex], dE_dw, this, weightIndex);
                }

                // Update Bias
                float dE_db = dE_do * do_dz;   // dz_dw = 1 for bias
                this.biases[oIdx] += backpropagationData.CalculateNewBias(this.biases[oIdx], dE_db, this, oIdx);   // allowed to change bias before full backpropagation because it has no effect in other layers directly
            }

            backpropagationData.dE_dz[this] = dE_dz;
            backpropagationData.UpdatedWeights[this] = newWeights;
        }

        internal override float CalculateErrorByOutputGradient(NNDetailedBackpropagationData detailedBackpropagationData, int fromNeuronIndex) {
            float[] dE_dz = detailedBackpropagationData.dE_dz[this];

            float dE_do = 0;
            for (int k = 0; k < Neurons; k++)
                dE_do += this.weights[ToWeightIndex(fromNeuronIndex, k)] * dE_dz[k];

            return dE_do;
        }

        private int ToWeightIndex(int fromNeuronIndex, int toNeuronIndex) => toNeuronIndex + fromNeuronIndex * Neurons;

        internal override void SetWeight(int weightIndex, float value) {
            this.weights[weightIndex] = value;
        }

        internal override int Weights => this.weights.Length;

        internal override void SetBias(int biasIndex, float value) {
            this.biases[biasIndex] = value;
        }

        internal override int Biases => this.biases.Length;

        internal override XMLElement ToXML() {
            XMLElement e = XMLElement.Create(GetType().Name);
            e.AddAttribute("Neurons", $"{Neurons}");
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
    }
}
