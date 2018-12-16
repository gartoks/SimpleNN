using System;
using System.Diagnostics;
using System.Text;
using SimpleNN.Data;
using SimpleNN.Layers;

namespace SimpleNN.Testing {
    public class Test_Convolution {

        internal static void Run() {
            //MaxPoolingTest();
            ConvolutionTest();
        }

        internal static void ConvolutionTest() {
            InputLayer iL = new InputLayer(3, 3, 1);

            ConvolutionalLayer cL = new ConvolutionalLayer(1, 2, 1, 0, iL, ActivationFunctions.None());
            //Debug.WriteLine("Filter ---------------");
            int i = 0;
            for (int y = 0; y < 2; y++) {
                StringBuilder sb = new StringBuilder();
                for (int x = 0; x < 2; x++, i++) {
                    if (x != 0)
                        sb.Append(", ");

                    float w = 0.1f * (i + 1);

                    cL.SetWeight(i, w);
                    sb.Append(w);
                }
                //Debug.WriteLine(sb.ToString());
            }

            FullyConnectedLayer fcL = new FullyConnectedLayer(1, cL, ActivationFunctions.None());
            //Debug.WriteLine("Weights ---------------");
            StringBuilder SB = new StringBuilder();
            for (int j = 0; j < fcL.Weights; j++, i++) {
                if (j != 0)
                    SB.Append(", ");

                float w = 1f;

                fcL.SetWeight(j, w);
                SB.Append(w);
            }
            //Debug.WriteLine(SB.ToString());


            NeuralNetwork nn = new NeuralNetwork(iL, cL, fcL);


            float[,,] inputValues = new float[3, 3, 1];
            i = 0;
            for (int z = 0; z < 1; z++) {
                //Debug.WriteLine($"{z} --------------------");
                for (int y = 0; y < 3; y++) {
                    StringBuilder sb = new StringBuilder();
                    for (int x = 0; x < 3; x++, i++) {
                        //if (x != 0)
                        //    sb.Append(",\t");

                        inputValues[x, y, z] = i + 1;
                        //sb.Append(inputValues[x, y, z]);
                    }
                    //Debug.WriteLine(sb.ToString());
                }
            }
            NNFeedData inputData = new NNFeedData(inputValues);
            Debug.WriteLine(nn.FeedForward(inputData)[0]);

            NNFeedData targetData = new NNFeedData(1, 1, 1, -3.2f);
            NNTrainingData trainingData = new NNPreloadedTrainingData(new[] {inputData}, new [] {targetData});

            NNBackpropagationData backpropagationData = new NNBackpropagationData(trainingData, 0.2f, (o, t) => o - t);

            //for (int j = 0; j < 1000; j++) {
                nn.PropagateBackward(backpropagationData);
            //}

            Debug.WriteLine(nn.FeedForward(inputData)[0]);

            //NNDetailedFeedData outputData = iL.FeedForward(inputData);
            //outputData = cL.FeedForward(outputData.OutputData);

            //Debug.WriteLine("Convolution Out");
            //for (int z = 0; z < cL.Depth; z++) {
            //    Debug.WriteLine($"{z} --------------------");
            //    for (int y = 0; y < cL.Height; y++) {
            //        StringBuilder sb = new StringBuilder();
            //        for (int x = 0; x < cL.Width; x++) {
            //            if (x != 0)
            //                sb.Append(", ");

            //            sb.Append($"{outputData.OutputData[outputData.OutputData.ToNeuronIndex(x, y, z)]}");
            //        }
            //        Debug.WriteLine(sb.ToString());
            //    }
            //}

            //outputData = fcL.FeedForward(outputData.OutputData);
            //Debug.WriteLine("Fully Connected Out");
            //Debug.WriteLine(outputData.OutputData[0]);

        }

        internal static void MaxPoolingTest() {
            InputLayer inputLayer = new InputLayer(20, 20, 2);

            MaxPoolingLayer mPLayer = new MaxPoolingLayer(2, 2, inputLayer);
            Debug.WriteLine(mPLayer.Width + " " + mPLayer.Height + " " + mPLayer.Depth);

            float[,,] inputValues = new float[20, 20, 2];
            int i = 0;
            for (int z = 0; z < 2; z++) {
                Debug.WriteLine($"{z} --------------------");
                for (int y = 0; y < 20; y++) {
                    StringBuilder sb = new StringBuilder();
                    for (int x = 0; x < 20; x++, i++) {
                        if (x != 0)
                            sb.Append(", ");

                        inputValues[x, y, z] = i;
                        sb.Append(i);
                    }
                    Debug.WriteLine(sb.ToString());
                }
            }
            NNFeedData inputData = new NNFeedData(inputValues);

            NNDetailedFeedData outputData = inputLayer.FeedForward(inputData);
            outputData = mPLayer.FeedForward(outputData.OutputData);

            for (int z = 0; z < mPLayer.Depth; z++) {
                Debug.WriteLine($"{z} --------------------");
                for (int y = 0; y < mPLayer.Height; y++) {
                    StringBuilder sb = new StringBuilder();
                    for (int x = 0; x < mPLayer.Width; x++) {
                        if (x != 0)
                            sb.Append(", ");

                        sb.Append($"{outputData.OutputData[outputData.OutputData.ToNeuronIndex(x, y, z)]}");
                    }
                    Debug.WriteLine(sb.ToString());
                }
            }
        }

    }
}