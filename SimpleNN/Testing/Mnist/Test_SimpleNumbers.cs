using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using SimpleNN.Data;
using SimpleNN.Layers;

namespace SimpleNN.Testing.Mnist {
    internal static class Test_SimpleNumbers {
        private const string file = @"..\numbers.bmp";
        private const string directory = @"..\SimpleNumbers\";

        private static List<MnistEntry> testing = new List<MnistEntry>();

        public static void Run() {
            Load();

            NeuralNetwork nn;

            bool load = true;
            if (load) {
                string loadPath = Path.Combine(directory, "nn.xml");
                string xml = File.ReadAllText(loadPath);
                nn = NeuralNetwork.FromXML(xml);
            } else {
                nn = InitializeNeuralNetwork(42);
                
                Train(nn, 0.35f, 100);
            }

            Console.WriteLine("Preparing Testing...");
            Test(nn);
            Console.WriteLine("Finished Testing.");

            if (!load && Console.ReadLine() == "store") {
                if (!Directory.Exists(directory))
                    Directory.CreateDirectory(directory);

                string fileName = $"{DateTime.UtcNow:dd-MM-yyyy-HH-mm-ss}.xml";
                string path = Path.Combine(directory, fileName);
                File.WriteAllText(path, nn.ToXML());
            }
        }

        private static void Test(NeuralNetwork nn, int tests = -1) {
            tests = tests == -1 ? testing.Count : tests;
            tests = Math.Min(testing.Count, tests);

            int score = 0;
            for (int i = 0; i < tests; i++) {
                Console.WriteLine($"Starting Test {i}/{tests}");

                MnistEntry entry = testing[i];

                NNFeedData inputData = new NNFeedData(3, 5, ConvertArray(entry.Image));
                NNFeedData outputData = nn.FeedForward(inputData);

                (int label, float value) guess = ArrayToLabel(outputData.CopyData());
                if (guess.label == entry.Label)
                    score++;

                Console.WriteLine($"{entry.Label} | {guess.label} ({guess.value:F2})");
            }

            Console.WriteLine($"{score}/{tests}");
        }

        private static void Train(NeuralNetwork nn, float learningRate, int runs) {
            Console.WriteLine("Preparing Training...");
            NNFeedData[] trainingInputData = new NNFeedData[testing.Count];
            NNFeedData[] trainingTargetData = new NNFeedData[testing.Count];
            for (int i = 0; i < testing.Count; i++) {
                MnistEntry entry = testing[i];

                trainingInputData[i] = new NNFeedData(3, 5, 1, ConvertArray(entry.Image));
                trainingTargetData[i] = new NNFeedData(10, 1, 1, LabelToArray(entry.Label));
            }
            NNTrainingData trainingData = new NNPreloadedTrainingData(trainingInputData, trainingTargetData);

            NNBackpropagationData backpropagationData = new NNBackpropagationData(trainingData, learningRate, (o, t) => o - t);

            double totalTime = 0;
            Console.WriteLine("Starting Training...");
            DateTime start = DateTime.UtcNow;
            for (int trainingRuns = 0; trainingRuns < runs; trainingRuns++) {
                int runs1 = trainingRuns;

                backpropagationData.BatchTrainingStartingCallback = (trainingDataIndex, trainingSets) => {
                    start = DateTime.UtcNow;
                };
                backpropagationData.BatchTrainingFinishedCallback = (trainingDataIndex, trainingSets) => {
                    totalTime += (DateTime.UtcNow - start).TotalMilliseconds;
                    double avgTime = totalTime / (trainingDataIndex + 1 + runs1 * trainingSets);
                    double remainingTime = avgTime * (trainingData.DataSize * (runs - runs1 + 1) - trainingDataIndex);
                    TimeSpan avgTSpan = TimeSpan.FromMilliseconds(avgTime);
                    TimeSpan remTSpan = TimeSpan.FromMilliseconds(remainingTime);
                    string avgTS = $"{avgTSpan:ss\\.ffff}";
                    string remTS = $"{remTSpan:hh\\:mm\\:ss}";

                    Console.WriteLine($"Finished Training {trainingDataIndex}/{trainingSets} RemainingTime: {remTS} AvgTime: {avgTS}");
                };

                nn.PropagateBackward(backpropagationData);
            }
        }

        private static NeuralNetwork InitializeNeuralNetwork(int seed) {
            Random random = new Random(seed == 0 ? new Random().Next() : seed);
            float RandomWeight() => (float)(random.NextDouble() * 2 - 1);

            Layer prevLayer;

            InputLayer li = new InputLayer(3, 5);
            prevLayer = li;

            ConvolutionalLayer l0 = new ConvolutionalLayer(8, 2, 1, 0, prevLayer, ActivationFunctions.ReLU(true));
            prevLayer = l0;
            prevLayer.InitializeWeights(RandomWeight);

            ConvolutionalLayer l2 = new ConvolutionalLayer(16, 2, 1, 0, prevLayer, ActivationFunctions.ReLU(true));
            prevLayer = l2;
            prevLayer.InitializeWeights(RandomWeight);

            FullyConnectedLayer l7 = new FullyConnectedLayer(16, prevLayer, ActivationFunctions.Sigmoid(1));
            prevLayer = l7;
            prevLayer.InitializeWeights(RandomWeight);

            FullyConnectedLayer l8 = new FullyConnectedLayer(10, prevLayer, ActivationFunctions.SoftMax(1));
            prevLayer = l8;
            prevLayer.InitializeWeights(RandomWeight);

            return new NeuralNetwork(li, l0, l2, l7, l8);
        }

        private static float[] ConvertArray(byte[] a) {
            float[] f = new float[a.Length];
            for (int j = 0; j < a.Length; j++)
                f[j] = a[j] / 255f;
            return f;
        }

        private static float[] LabelToArray(int label) {
            float[] o = new float[10];
            for (int i = 0; i < 10; i++) {
                o[i] = i == label ? 1 : 0;
            }

            return o;
        }

        private static (int label, float value) ArrayToLabel(float[] o) {
            float max = float.MinValue;
            int maxIdx = -1;

            for (int i = 0; i < o.Length; i++) {
                if (o[i] > max) {
                    max = o[i];
                    maxIdx = i;
                }
            }

            return (maxIdx, max);
        }

        private static void Load() {
            // Load data
            Console.WriteLine("Loading the dataset...");
            Bitmap bmp = new Bitmap(Test_SimpleNumbers.file);

            for (int numberIdx = 0; numberIdx < 10; numberIdx++) {
                byte[] data = new byte[3 * 5];
                for (int y = 0; y < 5; y++) {
                    for (int x = 0; x < 3; x++) {
                        Color color = bmp.GetPixel(numberIdx * 3 + x, y);
                        data[x + y * 3] = color.R;
                    }
                }

                MnistEntry entry = new MnistEntry(data, numberIdx);
                Test_SimpleNumbers.testing.Add(entry);
            }

            if (Test_SimpleNumbers.testing.Count == 0) {
                Console.WriteLine("Missing file.");
                Console.ReadKey();
                return;
            }
        }

    }
}
