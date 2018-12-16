using System;

namespace SimpleNN.Data {
    public sealed class NNFeedData {
        public int Width { get; }
        public int Height { get; }
        public int Depth { get; }
        private float[] Data { get; }

        public NNFeedData(int width, params float[] data)
            : this(width, 1, 1, data) { }


        public NNFeedData(int width, int height, params float[] data) 
            : this(width, height, 1, data) { }

        public NNFeedData(int width, int height, int depth, params float[] data) {
            if (data.Length != width * height * depth)
                throw new ArgumentException("The length of the data array must match the product of width, height and depth.", nameof(data));

            Width = width;
            Height = height;
            Depth = depth;
            Data = data;
        }

        public NNFeedData(float[,,] data) {
            Width = data.GetLength(0);
            Height = data.GetLength(1);
            Depth = data.GetLength(2);

            Data = new float[Width * Height * Depth];
            for (int z = 0; z < Depth; z++) {
                for (int y = 0; y < Height; y++) {
                    for (int x = 0; x < Width; x++) {
                        Data[ToNeuronIndex(x, y, z)] = data[x, y, z];
                    }
                }
            }
        }

        public float[] CopyData() {
            float[] ret = new float[Data.Length];
            Array.Copy(Data, 0, ret, 0, Data.Length);
            return ret;
        }

        public float this[int i] => Data[i];

        public int Size => Data.Length;

        internal int ToNeuronIndex(int x, int y, int z) => x + Width * (y + Height * z);
    }
}