using System.Collections.Generic;
using System.IO;
using System.Linq;
using CNTK;
using static CNTK.CNTKLib;

namespace CntkSerializerIssue
{
    class Program
    {
        const string FeaturesName = "features";
        const string LabelsName = "labels";
        const string TargetsName = "targets";
        const DataType DataTypeF32 = DataType.Float;
        static DeviceDescriptor Device = DeviceDescriptor.CPUDevice;

        static void Main(string[] args)
        {
            var repositoryRoot = @"..\..\..\..\..\";

            var channelNameToMapFilePath = new Dictionary<string, string>
            {
                { "Channel1", Path.Combine(repositoryRoot, @"mapfiles\TrainChannel1.map") },
                { "Channel2", Path.Combine(repositoryRoot, @"mapfiles\TrainChannel2.map") },
                { "Channel3", Path.Combine(repositoryRoot, @"mapfiles\TrainChannel3.map") },
                { "Channel4", Path.Combine(repositoryRoot, @"mapfiles\TrainChannel4.map") },
            };

            var ctfFilePath = Path.Combine(repositoryRoot, @"mapfiles\TrainTargets.ctf");
            var outputShape = 3;
            var maxSweeps = int.MaxValue;
            uint minibatchSize = 32;

            //
            // Input definition
            // 
            var channelInputShape = new int[] { 28, 28, 1 };

            var channelInputNDShape = NDShape.CreateNDShape(channelInputShape);
            var channelNameToInput = channelNameToMapFilePath.Keys
                .ToDictionary(n => n, n => Variable.InputVariable(channelInputNDShape, DataTypeF32, n));

            var channelInputsVector = new VariableVector(channelNameToInput.Values);
            // Splice input using channels -> (28, 28, 4).
            var input = Splice(channelInputsVector, new Axis(2), "MultiChannelInput");
            var inputScale = Constant.Scalar(DataTypeF32, 1.0 / byte.MaxValue, Device);
            Function inputNorm = ElementTimes(input, inputScale);

            // Network
            var network = LinearModel(inputNorm, outputShape);

            // Setup minibatch source
            var source = CreateTrainMinibatchSource(channelNameToMapFilePath, ctfFilePath, outputShape, maxSweeps);
            var channelNameToImageStreamInfo = channelNameToMapFilePath.ToDictionary(
                p => p.Key, p => source.StreamInfo(p.Key + FeaturesName));

            var trainTargetsStreamInfo = source.StreamInfo(TargetsName);

            var sweeps = 0;
            var dataDictionary = new Dictionary<Variable, MinibatchData>();

            // setup loss.
            Variable targets = Variable.InputVariable(new int[] { outputShape }, DataTypeF32);
            var loss = MeanSquareError(targets, network);

            // setup trainer.
            var learner = SGDLearner(new ParameterVector(network.Parameters().ToList()),
                new TrainingParameterScheduleDouble(0.001),
                new AdditionalLearningOptions());
            var trainer = CreateTrainer(network, loss, null, new LearnerVector() { learner });

            while (true)
            {
                var minibatchData = source.GetNextMinibatch(minibatchSize, Device);

                // Stop training once max epochs is reached.
                if (minibatchData.empty())
                {
                    System.Console.WriteLine($"Completed all {sweeps} sweeps");
                    break;
                }

                foreach (var channelName in channelNameToMapFilePath.Keys)
                {
                    var imageStreamInfo = channelNameToImageStreamInfo[channelName];
                    var dataInput = channelNameToInput[channelName];
                    var imageData = minibatchData[imageStreamInfo];
                    dataDictionary[dataInput] = imageData;
                }

                var targetsData = minibatchData[trainTargetsStreamInfo];
                dataDictionary[targets] = targetsData;

                trainer.TrainMinibatch(dataDictionary, Device);

                if (targetsData.sweepEnd)
                {
                    if (sweeps % 100 == 0)
                    {
                        System.Console.WriteLine($"Current sweep: {sweeps}. Loss: {trainer.PreviousMinibatchLossAverage()}");
                    }
                    sweeps++;
                }
            }

            System.Console.ReadKey();
        }


        static MinibatchSource CreateTrainMinibatchSource(
            IReadOnlyDictionary<string, string> channelNameToMapFilePath, string ctfFilePath,
            int outputShape, int maxEpochsOrSweeps)
        {
            var imageDeserializers = channelNameToMapFilePath.Select(p =>
            {
                var deserializer = CNTKLib.ImageDeserializer(p.Value, p.Key + LabelsName, (uint)1, p.Key + FeaturesName);
                AddGrayScaleTrue(deserializer);
                return deserializer;
            }).ToArray();

            var targetCtfDeserializer = CreateCtfDeserializer(ctfFilePath, outputShape);
            var deserializerList = new List<CNTKDictionary>(imageDeserializers)
            {
                targetCtfDeserializer
            };

            MinibatchSourceConfig config = new MinibatchSourceConfig(deserializerList)
            {
                MaxSweeps = (uint)maxEpochsOrSweeps,
            };

            return CNTKLib.CreateCompositeMinibatchSource(config);
        }

        static void AddGrayScaleTrue(CNTKDictionary deserializer)
        {
            deserializer.Add("grayscale", new DictionaryValue(true));
        }

        static CNTKDictionary CreateCtfDeserializer(string ctfFilePath, int outputShape)
        {
            var ctfStreamConfigurationVector = new StreamConfigurationVector();
            ctfStreamConfigurationVector.Add(new StreamConfiguration(TargetsName, outputShape, isSparse: false));

            var targetCtfDeserializer = CTFDeserializer(ctfFilePath, ctfStreamConfigurationVector);
            return targetCtfDeserializer;
        }

        static Function LinearModel(Variable input, int outputShape)
        {
            var inputFeatureCount = input.Shape.Dimensions.Aggregate((n, m) => n * m);
            var bias = new Parameter(new int[] { outputShape }, DataTypeF32, 0, Device);

            var weightInitializer = GlorotUniformInitializer(
                CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank, 1);

            var weights = new Parameter(new int[] { outputShape, inputFeatureCount },
                DataTypeF32, weightInitializer, Device);

            var flatten = Reshape(Flatten(input), new int[] { inputFeatureCount });
            return Plus(bias, CNTKLib.Times(weights, flatten));
        }

        static Function MeanSquareError(Variable targets, Variable predictions)
        {
            var errors = CNTKLib.Minus(targets, predictions);
            var squaredErrors = CNTKLib.Square(errors);
            var result = CNTKLib.ReduceMean(squaredErrors, new Axis(NDShape.InferredDimension));
            return result;
        }
    }
}
