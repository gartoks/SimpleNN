using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Net;
using SimpleNN.Data;
using SimpleNN.Layers;

namespace SimpleNN.Testing.Mnist {
    internal static class Test_Mnist {
        private const string urlMnist = @"http://yann.lecun.com/exdb/mnist/";
        private const string mnistFolder = @"..\Mnist\";
        private const string saveFolder = @"..\Mnist\Saves\";
        private const string trainingLabelFile = "train-labels-idx1-ubyte.gz";
        private const string trainingImageFile = "train-images-idx3-ubyte.gz";
        private const string testingLabelFile = "t10k-labels-idx1-ubyte.gz";
        private const string testingImageFile = "t10k-images-idx3-ubyte.gz";

        private static List<MnistEntry> testing;
        private static List<MnistEntry> training;

        public static void Run() {
            LoadMnist();

            //NeuralNetwork nn = InitializeNeuralNetwork(42);

            string loadPath = Path.Combine(saveFolder, "primary_30000.xml");
            string xml = File.ReadAllText(loadPath);
            NeuralNetwork nn = NeuralNetwork.FromXML(xml);

            //Train(nn, 0.35f, 1);

            Console.WriteLine("Preparing Testing...");
            Test(nn, 10);
            Console.WriteLine("Finished Testing.");
        }

        private static void Test(NeuralNetwork nn, int tests = -1) {
            tests = tests == -1 ? testing.Count : tests;
            tests = Math.Min(testing.Count, tests);

            int score = 0;
            for (int i = 0; i < tests; i++) {
                Console.WriteLine($"Starting Test {i}/{tests}");

                MnistEntry entry = testing[i];

                NNFeedData inputData = new NNFeedData(28, 28, ConvertArray(entry.Image));
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
            NNFeedData[] trainingInputData = new NNFeedData[training.Count];
            NNFeedData[] trainingTargetData = new NNFeedData[training.Count];
            for (int i = 0; i < training.Count; i++) {
                MnistEntry entry = training[i];

                trainingInputData[i] = new NNFeedData(28, 28, 1, ConvertArray(entry.Image));
                trainingTargetData[i] = new NNFeedData(10, 1, 1, LabelToArray(entry.Label));
            }
            NNTrainingData trainingData = new NNPreloadedTrainingData(trainingInputData, trainingTargetData);

            NNBackpropagationData backpropagationData = new NNBackpropagationData(trainingData, learningRate, (o, t) => o - t);

            double totalTime = 0;
            Console.WriteLine("Starting Training...");
            for (int trainingRuns = 0; trainingRuns < runs; trainingRuns++) {
                DateTime start = DateTime.UtcNow;

                backpropagationData.BatchTrainingStartingCallback = (trainingDataIndex, trainingSets) => {
                    start = DateTime.UtcNow;
                };
                backpropagationData.BatchTrainingFinishedCallback = (trainingDataIndex, trainingSets) => {
                    totalTime += (DateTime.UtcNow - start).TotalMilliseconds;
                    double avgTime = totalTime / (trainingDataIndex + 1);
                    double remainingTime = avgTime * (trainingSets - trainingDataIndex);
                    TimeSpan avgTSpan = TimeSpan.FromMilliseconds(avgTime);
                    TimeSpan remTSpan = TimeSpan.FromMilliseconds(remainingTime);
                    TimeSpan totTSpan = TimeSpan.FromMilliseconds(totalTime);
                    string avgTS = $"{avgTSpan:ss\\.ffff}";
                    string remTS = $"{remTSpan:hh\\:mm\\:ss}";
                    string totTS = $"{totTSpan:hh\\:mm\\:ss}";

                    Save(nn, $"primary_{trainingDataIndex}");

                    Console.WriteLine($"Finished Training {trainingDataIndex}/{trainingSets} Passed:{totTS} Remaining:{remTS} Avg:{avgTS}");
                };


                nn.PropagateBackward(backpropagationData);
            }
        }

        private static NeuralNetwork InitializeNeuralNetwork(int seed) {
            Random random = new Random(seed == 0 ? new Random().Next() : seed);
            float RandomWeight() => (float)(random.NextDouble() * 2 - 1);

            Layer prevLayer;

            InputLayer li = new InputLayer(28, 28);
            prevLayer = li;

            ConvolutionalLayer l0 = new ConvolutionalLayer(15, 5, 1, 0, prevLayer, ActivationFunctions.ReLU(true));
            prevLayer = l0;
            prevLayer.InitializeWeights(RandomWeight);

            MaxPoolingLayer l1 = new MaxPoolingLayer(2, 2, prevLayer);
            prevLayer = l1;

            ConvolutionalLayer l2 = new ConvolutionalLayer(30, 4, 1, 0, prevLayer, ActivationFunctions.ReLU(true));
            prevLayer = l2;
            prevLayer.InitializeWeights(RandomWeight);

            MaxPoolingLayer l3 = new MaxPoolingLayer(3, 2, prevLayer);
            prevLayer = l3;

            ConvolutionalLayer l4 = new ConvolutionalLayer(45, 2, 2, 0, prevLayer, ActivationFunctions.ReLU(true));
            prevLayer = l4;
            prevLayer.InitializeWeights(RandomWeight);

            MaxPoolingLayer l5 = new MaxPoolingLayer(2, 1, prevLayer);
            prevLayer = l5;

            FullyConnectedLayer l6 = new FullyConnectedLayer(64, prevLayer, ActivationFunctions.Sigmoid(1));
            prevLayer = l6;
            prevLayer.InitializeWeights(RandomWeight);

            FullyConnectedLayer l7 = new FullyConnectedLayer(32, prevLayer, ActivationFunctions.Sigmoid(1));
            prevLayer = l7;
            prevLayer.InitializeWeights(RandomWeight);

            FullyConnectedLayer l8 = new FullyConnectedLayer(10, prevLayer, ActivationFunctions.SoftMax(1));
            prevLayer = l8;
            prevLayer.InitializeWeights(RandomWeight);

            return new NeuralNetwork(li, l0, l1, l2, l3, l4, l5, l6, l7, l8);
        }

        private static void Save(NeuralNetwork nn, string fileName) {
            if (!Directory.Exists(saveFolder))
                Directory.CreateDirectory(saveFolder);

            foreach (FileInfo file in new DirectoryInfo(saveFolder).EnumerateFiles()) {
                file.Delete();
            }

            string path = Path.Combine(saveFolder, fileName + ".xml");
            File.WriteAllText(path, nn.ToXML());
        }

        private static float[] ConvertArray(byte[] a) {
            float[] f = new float[a.Length];
            for (int j = 0; j < a.Length; j++)
                f[j] = a[j];
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

        private static void LoadMnist() {
            Directory.CreateDirectory(mnistFolder);

            string trainingLabelFilePath = Path.Combine(mnistFolder, trainingLabelFile);
            string trainingImageFilePath = Path.Combine(mnistFolder, trainingImageFile);
            string testingLabelFilePath = Path.Combine(mnistFolder, testingLabelFile);
            string testingImageFilePath = Path.Combine(mnistFolder, testingImageFile);

            // Download Mnist files if needed
            Console.WriteLine("Downloading Mnist training files...");
            DownloadFile(urlMnist + trainingLabelFile, trainingLabelFilePath);
            DownloadFile(urlMnist + trainingImageFile, trainingImageFilePath);
            Console.WriteLine("Downloading Mnist testing files...");
            DownloadFile(urlMnist + testingLabelFile, testingLabelFilePath);
            DownloadFile(urlMnist + testingImageFile, testingImageFilePath);

            // Load data
            Console.WriteLine("Loading the datasets...");
            Test_Mnist.training = MnistReader.Load(trainingLabelFilePath, trainingImageFilePath);
            Test_Mnist.testing = MnistReader.Load(testingLabelFilePath, testingImageFilePath);

            //ExtractImages(Test_Mnist.testing);

            if (Test_Mnist.training.Count == 0 || Test_Mnist.testing.Count == 0) {
                Console.WriteLine("Missing Mnist training/testing files.");
                Console.ReadKey();
                return;
            }
        }

        private static void ExtractImages(IEnumerable<MnistEntry> entries) {
            Directory.CreateDirectory(@"..\Mnist\Extracted\");

            int i = 0;
            foreach (MnistEntry mnistEntry in entries) {
                Console.WriteLine(i);

                Bitmap bmp = new Bitmap(28, 28);
                for (int y = 0; y < 28; y++) {
                    for (int x = 0; x < 28; x++) {
                        int c = mnistEntry.Image[x + y * 28];
                        Color color = Color.FromArgb(c, c, c);
                        bmp.SetPixel(x, y, color);
                    }
                }

                bmp.Save($"..\\Mnist\\Extracted\\{i}_{mnistEntry.Label}.png", ImageFormat.Png);
                i++;
            }
        }

        private static void DownloadFile(string urlFile, string destFilepath) {
            if (!File.Exists(destFilepath)) {
                try {
                    using (var client = new WebClient()) {
                        client.DownloadFile(urlFile, destFilepath);
                    }
                } catch (Exception e) {
                    Console.WriteLine("Failed downloading " + urlFile);
                    Console.WriteLine(e.Message);
                }
            }
        }

    }
}
