using System;
using SimpleNN.Data;
using SimpleNN.Layers;

namespace SimpleNN.Testing {
    internal class Test_Sin {

        internal static void Run() {
            NeuralNetwork nn = InitializeNeuralNetwork(42, ActivationFunctions.Sigmoid(1));

            Calculate(nn);

            Train(nn, 0.4f, 100000);

            Calculate(nn);
        }

        private static void Calculate(NeuralNetwork nn) {
            float error = 0;

            Random r = new Random(42);

            for (int i = 0; i < 10; i++) {
                float x = (float)(2.0 * Math.PI * r.NextDouble());

                NNFeedData inputData = new NNFeedData(1, new float[] { x });
                NNFeedData outputData = nn.FeedForward(inputData);

                float t = (float)Math.Sin(x) / 2f + 0.5f;
                error += (float)Math.Abs(outputData[0] - t);

                Console.WriteLine($"sin({x/Math.PI*180f:F1}) => {Math.Sin(x):F2}\t| {(outputData[0] * 2f - 1f):F2}");
            }

            Console.WriteLine(error / 10f);
        }

        private static NeuralNetwork InitializeNeuralNetwork(int seed, ActivationFunctions.ActivationFunction activationFunction) {
            Random random = new Random(seed == 0 ? new Random().Next() : seed);
            float RandomWeight() => (float)(random.NextDouble() * 2 - 1);

            InputLayer l0 = new InputLayer(1);

            FullyConnectedLayer l1 = new FullyConnectedLayer(6, l0, activationFunction);
            for (int i = 0; i < l1.Weights; i++) {
                l1.SetWeight(i, RandomWeight() * 0.25f);
            }

            FullyConnectedLayer l2 = new FullyConnectedLayer(1, l1, activationFunction);
            for (int i = 0; i < l2.Weights; i++) {
                l2.SetWeight(i, RandomWeight() * 0.25f);
            }
            return new NeuralNetwork(l0, l1, l2);
        }

        private static void Train(NeuralNetwork nn, float learningRate, int runs) {
            NNFeedData[] trainingInputData = new NNFeedData[360];
            NNFeedData[] trainingTargetData = new NNFeedData[360];
            for (int i = 0; i < 360; i++) {
                float x = i * (float)Math.PI / 180f;

                trainingInputData[i] = new NNFeedData(1, height: 1, depth: 1, data: new float[] { x });
                trainingTargetData[i] = new NNFeedData(1, height: 1, depth: 1, data: new float[] { (float)Math.Sin(x) / 2f + 0.5f });
            }
            NNTrainingData trainingData = new NNPreloadedTrainingData(trainingInputData, trainingTargetData);

            NNBackpropagationData backpropagationData = new NNBackpropagationData(trainingData, learningRate, (o, t) => o - t);

            for (int trainingRuns = 0; trainingRuns < runs; trainingRuns++) {
                nn.PropagateBackward(backpropagationData);
            }
        }


    }
}