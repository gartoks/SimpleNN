using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SimpleNN;
using SimpleNN.Data;
using SimpleNN.Layers;
using SimpleXML;

namespace SimpleNNTests {
    [TestClass]
    public class MaxPoolingLayerTest {

        [TestMethod]
        public void TestConstructor() {
            LayerStub prevLayer = new LayerStub(4, 4, 2);

            MaxPoolingLayer layer = new MaxPoolingLayer(2, 2, prevLayer);

            Assert.AreEqual(2, layer.Width);
            Assert.AreEqual(2, layer.Height);
            Assert.AreEqual(prevLayer.Depth, layer.Depth);
        }

        [TestMethod]
        public void TestFeedForward() {
            float[] data = new float[4 * 4 * 2];
            for (int i = 0; i < data.Length; i++) {
                data[i] = i + 1;
            }
            LayerStub prevLayer = new LayerStub(4, 4, 2);
            MaxPoolingLayer layer = new MaxPoolingLayer(2, 2, prevLayer);
            NNFeedData feedData = new NNFeedData(4, 4, 2, data);

            NNDetailedFeedData result = layer.FeedForward(feedData);

            Assert.AreEqual(6, result.OutputData[result.OutputData.ToNeuronIndex(0, 0, 0)]);
            Assert.AreEqual(8, result.OutputData[result.OutputData.ToNeuronIndex(1, 0, 0)]);
            Assert.AreEqual(14, result.OutputData[result.OutputData.ToNeuronIndex(0, 1, 0)]);
            Assert.AreEqual(16, result.OutputData[result.OutputData.ToNeuronIndex(1, 1, 0)]);
        }

        [TestMethod]
        public void TestCalculateErrorByOutputGradient() {
            LayerStub prevLayer = new LayerStub(4, 4, 2);
            MaxPoolingLayer layer = new MaxPoolingLayer(2, 2, prevLayer);
            
            
            
            //PrivateObject obj = new PrivateObject(layer);
            //int[] maxIndices = new[] { 5, 7, 13, 15 };

            //for (int fNIdx = 0; fNIdx < prevLayer.Neurons; fNIdx++) {
            //    for (int tNIdx = 0; tNIdx < layer.Neurons; tNIdx++) {
            //        float weight = (float)obj.Invoke("GetWeight", maxIndices, fNIdx, tNIdx);


            //    }
            //}
        }

        private class LayerStub : Layer {
            public LayerStub(int width, int height, int depth)
                : base(width, height, depth) { }

            public LayerStub(int width, int height, int depth, Layer previousLayer)
                : base(width, height, depth, previousLayer) { }

            internal override NNDetailedFeedData FeedForward(NNFeedData input) {
                return null;
            }

            internal override void PropagateBackward(NNDetailedBackpropagationData backpropagationData) {
            }

            internal override float CalculateErrorByOutputGradient(NNDetailedBackpropagationData detailedBackpropagationData, int fromNeuronIndex) {
                return 0;
            }

            internal override void SetWeight(int weightIndex, float value) {
            }

            internal override int Weights => 0;

            internal override void SetBias(int biasIndex, float value) {
            }

            internal override int Biases => 0;

            internal override XMLElement ToXML() {
                return null;
            }
        }

    }
}
