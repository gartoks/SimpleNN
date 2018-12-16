namespace SimpleNN.Testing.Mnist {
    internal class MnistEntry {
        public readonly byte[] Image;
        public readonly int Label;

        public MnistEntry(byte[] image, int label) {
            Image = image;
            Label = label;
        }

        public override string ToString() {
            return $"Label: {Label}";
        }
    }
}