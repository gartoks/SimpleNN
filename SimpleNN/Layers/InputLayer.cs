using System;
using SimpleNN.Data;
using SimpleXML;

namespace SimpleNN.Layers {
    public class InputLayer : Layer {
        public InputLayer(int width, int height = 1, int depth = 1)
            : base(width, height, depth) {
        }

        internal override NNDetailedFeedData FeedForward(NNFeedData input) {
            if (input.Width != Width || input.Height != Height || input.Depth != Depth)
                throw new ArgumentOutOfRangeException(nameof(input));

            float[] output = input.CopyData();
            return new NNDetailedFeedData(this, output, output);
        }

        internal override void PropagateBackward(NNDetailedBackpropagationData backpropagationData) {
            // Do nothing
        }

        internal override float CalculateErrorByOutputGradient(NNDetailedBackpropagationData detailedBackpropagationData, int fromNeuronIndex) {
            throw new NotImplementedException("Is never used.");
        }

        internal override void SetWeight(int weightIndex, float value) {
            // Do nothing
        }

        internal override int Weights => 0;

        internal override void SetBias(int biasIndex, float value) {
            // Do nothing
        }

        internal override int Biases => 0;

        internal override XMLElement ToXML() {
            XMLElement e = XMLElement.Create(GetType().Name);
            e.AddAttribute("Width", $"{Width}");
            e.AddAttribute("Height", $"{Height}");
            e.AddAttribute("Depth", $"{Depth}");
            return e;
        }
    }
}