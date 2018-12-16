using System;
using System.Collections.Generic;

namespace SimpleNN.Data {
    internal class NNDetailedBackpropagationData {

        internal readonly NNBackpropagationData BackpropagationData;
        private int trainingDataIndex;

        internal readonly Dictionary<Layer, Dictionary<int, float>>[] weightChanges;
        internal readonly Dictionary<Layer, Dictionary<int, float>>[] biasChanges;

        internal readonly Dictionary<Layer, NNDetailedFeedData> FeedForwardData;
        internal readonly Dictionary<Layer, float[]> dE_dz;
        internal readonly Dictionary<Layer, float[]> UpdatedWeights;

        public NNDetailedBackpropagationData(NNBackpropagationData backpropagationData, int trainingDataSets) {
            BackpropagationData = backpropagationData;

            this.weightChanges = new Dictionary<Layer, Dictionary<int, float>>[trainingDataSets];
            this.biasChanges = new Dictionary<Layer, Dictionary<int, float>>[trainingDataSets];

            FeedForwardData = new Dictionary<Layer, NNDetailedFeedData>();
            dE_dz = new Dictionary<Layer, float[]>();
            UpdatedWeights = new Dictionary<Layer, float[]>();
        }

        internal float ErrorGradient(float output, int outputNeuronIndex) {
            return BackpropagationData.ErrorGradient(output, BackpropagationData.TrainingData.GetTargetData(TrainingDataIndex)[outputNeuronIndex]);
        }

        internal float CalculateNewBias(float bias, float dE_db, Layer layer, int biasIndex) {
            float learningMomentum = 0;
            if (TrainingDataIndex > 0)
                learningMomentum = BackpropagationData.LearningMomentum * biasChanges[trainingDataIndex - 1][layer][biasIndex];

            float biasChange = BackpropagationData.LearningRate * dE_db + learningMomentum;

            this.biasChanges[trainingDataIndex][layer][biasIndex] = biasChange;

            return bias - biasChange;
        }

        internal float CalculateNewWeight(float weight, float dE_dw, Layer layer, int weightIndex) {
            float learningMomentum = 0;
            if (TrainingDataIndex > 0)
                learningMomentum = BackpropagationData.LearningMomentum * weightChanges[trainingDataIndex - 1][layer][weightIndex];

            float weightChange = BackpropagationData.LearningRate * dE_dw + learningMomentum;

            this.weightChanges[trainingDataIndex][layer][weightIndex] = weightChange;

            return weight - weightChange;
        }

        internal int TrainingDataIndex {
            get => this.trainingDataIndex;
            set {
                if (value < 0 || value >= this.weightChanges.Length)
                    throw new ArgumentOutOfRangeException(nameof(value), "The training data index must be in [0, trainingDataSets].");

                this.trainingDataIndex = value;
            }
        }

    }
}