using System;

namespace SimpleNN.Data {
    public sealed class NNBackpropagationData {

        public NNTrainingData TrainingData { get; }
        private float learningRate;
        private float learningMomentum;
        private Func<float, float, float> errorGradient;

        public Action<int, int> BatchTrainingStartingCallback { get; set; } 
        public Action<int, int> BatchTrainingFinishedCallback { get; set; }

        public NNBackpropagationData(NNTrainingData trainingData, float learningRate, Func<float, float, float> errorGradient) {
            TrainingData = trainingData;
            this.learningRate = learningRate;
            this.errorGradient = errorGradient;
        }

        public float LearningRate {
            get => this.learningRate;
            set {
                if (value <= 0 || value >= 1)
                    throw new ArgumentOutOfRangeException(nameof(value), "The learning rate must be in ]0, 1[.");

                this.learningRate = value;
            }
        }

        public float LearningMomentum {
            get => this.learningMomentum;
            set {
                if (value < 0 || value >= 1)
                    throw new ArgumentOutOfRangeException(nameof(value), "The momentum rate must be in [0, 1[.");

                this.learningMomentum = value;
            }
        }

        public Func<float, float, float> ErrorGradient {
            get => this.errorGradient;
            set => this.errorGradient = value ?? throw new ArgumentNullException(nameof(value));
        }

    }
}