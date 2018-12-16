using System;
using System.Collections.Generic;
using System.Linq;
using SimpleXML;

namespace SimpleNN {
    public static class ActivationFunctions {
        internal static ActivationFunction FromXML(XMLElement root) {
            if (root.GetAttribute("Name").Value == "None") {
                return None();
            }

            if (root.GetAttribute("Name").Value == "Sigmoid") {
                XMLElement customDataXE = root.ChildWithTag("CustomData");
                float steepness = int.Parse(customDataXE.ChildWithTag("steepness").GetAttribute("Value").Value);

                return Sigmoid(steepness);
            }

            if (root.GetAttribute("Name").Value == "ReLU") {
                XMLElement customDataXE = root.ChildWithTag("CustomData");
                bool zeroIsPositive = bool.Parse(customDataXE.ChildWithTag("zeroIsPositive").GetAttribute("Value").Value);

                return ReLU(zeroIsPositive);
            }

            if (root.GetAttribute("Name").Value == "TanH") {
                XMLElement customDataXE = root.ChildWithTag("CustomData");
                float steepness = int.Parse(customDataXE.ChildWithTag("steepness").GetAttribute("Value").Value);

                return TanH(steepness);
            }

            if (root.GetAttribute("Name").Value == "LeakyReLU") {
                XMLElement customDataXE = root.ChildWithTag("CustomData");
                float leakynessFactor = int.Parse(customDataXE.ChildWithTag("leakynessFactor").GetAttribute("Value").Value);
                bool zeroIsPositive = bool.Parse(customDataXE.ChildWithTag("zeroIsPositive").GetAttribute("Value").Value);

                return LeakyReLU(leakynessFactor, zeroIsPositive);
            }

            if (root.GetAttribute("Name").Value == "SoftPlus") {
                XMLElement customDataXE = root.ChildWithTag("CustomData");
                float steepness = int.Parse(customDataXE.ChildWithTag("steepness").GetAttribute("Value").Value);

                return SoftPlus(steepness);
            }

            if (root.GetAttribute("Name").Value == "SoftMax") {
                XMLElement customDataXE = root.ChildWithTag("CustomData");
                float steepness = int.Parse(customDataXE.ChildWithTag("steepness").GetAttribute("Value").Value);

                return SoftMax(steepness);
            }

            throw new NotImplementedException();    // Never reached
        }

        public class ActivationFunction {
            public readonly string Name;
            public readonly Activation Function;
            public readonly Activation Gradient;

            private readonly Dictionary<string, (Type t, object val)> customData;

            public ActivationFunction(string name, Activation function, Activation gradient) {
                Name = name;
                Function = function;
                Gradient = gradient;
                this.customData = new Dictionary<string, (Type t, object val)>();
            }

            public void Set<T>(string key, T value) {
                this.customData[key] = (typeof(T), value);
            }

            public bool Get<T>(string key, out T value) {
                value = default(T);

                if (!this.customData.TryGetValue(key, out (Type t, object val) tuple))
                    return false;

                value = (T)tuple.val;
                return true;
            }

            internal XMLElement ToXML() {
                XMLElement e = XMLElement.Create("ActivationFunction");
                e.AddAttribute("Name", Name);

                XMLElement customDataXE = XMLElement.Create("CustomData");
                e.AddChild(customDataXE);
                foreach (KeyValuePair<string, (Type t, object val)> tuple in this.customData) {
                    XMLElement tupleXE = XMLElement.Create(tuple.Key);
                    tupleXE.AddAttribute("Type", tuple.Value.t.FullName);
                    tupleXE.AddAttribute("Value", tuple.Value.val.ToString());

                    customDataXE.AddChild(tupleXE);
                }

                return e;
            }
        }

        public delegate float Activation(float output, float[] outputs);

        public static ActivationFunction None() {
            Activation f = (x, xs) => x;
            Activation df_dx = (x, xs) => 1;

            return new ActivationFunction("None", f, df_dx);
        }

        public static ActivationFunction Sigmoid(float steepness) {
            Activation f = (x, xs) => 1.0f / (1.0f + (float)Math.Exp(-steepness * x));
            Activation df_dx = (x, xs) => f(x, xs) * (1f - f(x, xs));

            ActivationFunction a = new ActivationFunction("Sigmoid", f, df_dx);
            a.Set("steepness", steepness);

            return a;
        }

        public static ActivationFunction ReLU(bool zeroIsPositive) {
            Activation f = (x, xs) => x < 0 || (x == 0 && !zeroIsPositive) ? 0 : x;
            Activation df_dx = (x, xs) => x < 0 || (x == 0 && !zeroIsPositive) ? 0 : 1;

            ActivationFunction a = new ActivationFunction("ReLU", f, df_dx);
            a.Set("zeroIsPositive", zeroIsPositive);

            return a;
        }

        public static ActivationFunction TanH(float steepness) {
            Activation f = (x, xs) => (float)Math.Tanh(steepness * x);
            Activation df_dx = (x, xs) => 1f - f(x, xs) * f(x, xs);

            ActivationFunction a = new ActivationFunction("TanH", f, df_dx);
            a.Set("steepness", steepness);

            return a;
        }

        public static ActivationFunction LeakyReLU(float leakynessFactor, bool zeroIsPositive) {
            Activation f = (x, xs) => x < 0 || (x == 0 && !zeroIsPositive) ? leakynessFactor * x : x;
            Activation df_dx = (x, xs) => x < 0 || (x == 0 && !zeroIsPositive) ? leakynessFactor : 1;

            ActivationFunction a = new ActivationFunction("LeakyReLU", f, df_dx);
            a.Set("leakynessFactor", leakynessFactor);
            a.Set("zeroIsPositive", zeroIsPositive);

            return a;
        }

        public static ActivationFunction SoftPlus(float steepness) {
            Activation f = (x, xs) => (float)Math.Log(1.0 + Math.Exp(steepness * x));
            Activation df_dx = (x, xs) => steepness - steepness / (1f + (float)Math.Exp(steepness * x));

            ActivationFunction a = new ActivationFunction("SoftPlus", f, df_dx);
            a.Set("steepness", steepness);

            return a;
        }

        public static ActivationFunction SoftMax(float steepness) {
            Activation f = (x, xs) => (float)(Math.Exp(x) / xs.Sum(t => Math.Exp(t)));
            Activation df_dx = (x, xs) => f(x, xs) * (1f - f(x, xs));

            ActivationFunction a = new ActivationFunction("SoftMax", f, df_dx);
            a.Set("steepness", steepness);

            return a;
        }
    }
}