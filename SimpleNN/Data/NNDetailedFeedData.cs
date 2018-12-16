using System.Collections.Generic;

namespace SimpleNN.Data {
    internal class NNDetailedFeedData {
        public readonly Layer Layer;

        public readonly NNFeedData OutputData;
        public readonly float[] RawOutputData;
        public readonly Dictionary<string, object> CustomData;

        public NNDetailedFeedData(Layer layer, float[] outputData, float[] rawOutputData) {
            Layer = layer;
            OutputData = new NNFeedData(layer.Width, layer.Height, layer.Depth, outputData);
            RawOutputData = rawOutputData;
            CustomData = new Dictionary<string, object>();
        }
    }
}