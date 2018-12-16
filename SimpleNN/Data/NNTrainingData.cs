using System;

namespace SimpleNN.Data {
    public abstract class NNTrainingData {
        public abstract NNFeedData GetInputData(int index);

        public abstract NNFeedData GetTargetData(int index);

        public abstract int DataSize { get; }

        public abstract int InputDataWidth { get; }

        public abstract int InputDataHeight { get; }

        public abstract int InputDataDepth { get; }

        public abstract int TargetDataWidth { get; }

        public abstract int TargetDataHeight { get; }

        public abstract int TargetDataDepth { get; }
    }

    public sealed class NNOnDemandTrainingData : NNTrainingData {
        private readonly Func<int, NNFeedData> loadInputData;
        private readonly Func<int, NNFeedData> loadTargetData;

        public NNOnDemandTrainingData(
            Func<int, NNFeedData> loadInputData,
            Func<int, NNFeedData> loadTargetData,
            int dataSize,
            (int width, int height, int depth) inputDataSize,
            (int width, int height, int depth) targetDataSize) {

            DataSize = dataSize;

            InputDataWidth = inputDataSize.width;
            InputDataHeight = inputDataSize.height;
            InputDataDepth = inputDataSize.depth;

            TargetDataWidth = targetDataSize.width;
            TargetDataHeight = targetDataSize.height;
            TargetDataDepth = targetDataSize.depth;

            this.loadInputData = loadInputData ?? throw new ArgumentNullException(nameof(loadInputData));
            this.loadTargetData = loadTargetData ?? throw new ArgumentNullException(nameof(loadTargetData));
        }

        public override NNFeedData GetInputData(int index) {
            NNFeedData data = loadInputData(index);

            if (data.Width != InputDataWidth)
                throw new ArgumentException("Loaded input data width does not match training data.");

            if (data.Height != InputDataHeight)
                throw new ArgumentException("Loaded input data height does not match training data.");

            if (data.Depth != InputDataDepth)
                throw new ArgumentException("Loaded input data depth does not match training data.");

            return data;
        }

        public override NNFeedData GetTargetData(int index) {
            NNFeedData data = loadTargetData(index);

            if (data.Width != TargetDataWidth)
                throw new ArgumentException("Loaded target data width does not match training data.");

            if (data.Height != TargetDataHeight)
                throw new ArgumentException("Loaded target data height does not match training data.");

            if (data.Depth != TargetDataDepth)
                throw new ArgumentException("Loaded target data depth does not match training data.");

            return data;
        }

        public override int DataSize { get; }

        public override int InputDataWidth { get; }

        public override int InputDataHeight { get; }

        public override int InputDataDepth { get; }

        public override int TargetDataWidth { get; }

        public override int TargetDataHeight { get; }

        public override int TargetDataDepth { get; }
    }

    public sealed class NNPreloadedTrainingData : NNTrainingData {
        private readonly NNFeedData[] inputData;
        private readonly NNFeedData[] targetData;

        public NNPreloadedTrainingData(NNFeedData[] inputData, NNFeedData[] targetData) {
            if (inputData.Length != targetData.Length)
                throw new ArgumentException("Input and target data must have the same count.");

            int iW = inputData[0].Width;
            int iH = inputData[0].Height;
            int iD = inputData[0].Depth;

            int tW = targetData[0].Width;
            int tH = targetData[0].Height;
            int tD = targetData[0].Depth;
            for (int i = 0; i < inputData.Length; i++) {
                if (inputData[i].Width != iW || inputData[i].Height != iH || inputData[i].Depth != iD)
                    throw new ArgumentException();

                if (targetData[i].Width != tW || targetData[i].Height != tH || targetData[i].Depth != tD)
                    throw new ArgumentException();
            }

            this.inputData = inputData;
            this.targetData = targetData;
        }

        public override NNFeedData GetInputData(int index) => this.inputData[index];

        public override NNFeedData GetTargetData(int index) => this.targetData[index];

        public override int DataSize => this.inputData.Length;

        public override int InputDataWidth => this.inputData[0].Width;

        public override int InputDataHeight => this.inputData[0].Height;

        public override int InputDataDepth => this.inputData[0].Depth;

        public override int TargetDataWidth => this.targetData[0].Width;

        public override int TargetDataHeight => this.targetData[0].Height;

        public override int TargetDataDepth => this.targetData[0].Depth;
    }
}