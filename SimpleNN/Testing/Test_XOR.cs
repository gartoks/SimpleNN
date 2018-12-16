using System;
using SimpleNN.Data;
using SimpleNN.Layers;

namespace SimpleNN.Testing {
    internal class Test_XOR {

        internal static void Run() {
            NeuralNetwork nn = InitializeNeuralNetwork(0, ActivationFunctions.Sigmoid(1));

            CalculateXOR(nn);

            TrainXOR(nn, 0.5f, 1000000);

            CalculateXOR(nn);
        }

        private static void CalculateXOR(NeuralNetwork nn) {
            float score = 0;
            for (int i = 0; i < Math.Pow(2, 2); i++) {
                int a = i % 2;
                int b = i / 2;

                int r = a ^ b;

                NNFeedData inputData = new NNFeedData(2, new float[] { a, b });
                NNFeedData outputData = nn.FeedForward(inputData);
                score += (float)Math.Abs(outputData[0] - r);

                Console.WriteLine($"{a} ^ {b} => {r} | {outputData[0]}");
            }

            Console.WriteLine(score / 4);
        }

        private static NeuralNetwork InitializeNeuralNetwork(int seed, ActivationFunctions.ActivationFunction activationFunction) {
            Random random = new Random(seed == 0 ? new Random().Next() : seed);
            float RandomWeight() => (float)(random.NextDouble() * 2 - 1);

            InputLayer l0 = new InputLayer(2);

            FullyConnectedLayer l1 = new FullyConnectedLayer(2, l0, activationFunction);
            for (int i = 0; i < l1.Weights; i++) {
                l1.SetWeight(i, RandomWeight() * 0.25f);
            }

            FullyConnectedLayer l2 = new FullyConnectedLayer(1, l1, activationFunction);
            for (int i = 0; i < l2.Weights; i++) {
                l2.SetWeight(i, RandomWeight() * 0.25f);
            }
            return new NeuralNetwork(l0, l1, l2);
        }

        private static void TrainXOR(NeuralNetwork nn, float learningRate, int runs) {
            NNFeedData[] trainingInputData = new NNFeedData[(int)Math.Pow(2, 2)];
            NNFeedData[] trainingTargetData = new NNFeedData[(int)Math.Pow(2, 2)];
            for (int i = 0; i < Math.Pow(2, 2); i++) {
                int a = i % 2;
                int b = i / 2;
                int r = a ^ b;

                trainingInputData[i] = new NNFeedData(2, 1, 1, a, b);
                trainingTargetData[i] = new NNFeedData(1, 1, 1, r);
            }
            NNTrainingData trainingData = new NNPreloadedTrainingData(trainingInputData, trainingTargetData);

            NNBackpropagationData backpropagationData = new NNBackpropagationData(trainingData, learningRate, (o, t) => o - t);

            for (int trainingRuns = 0; trainingRuns < runs; trainingRuns++) {
                nn.PropagateBackward(backpropagationData);
            }
        }


    }
}