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

        static void Main(string[] args)
        {
            var d = DeviceDescriptor.CPUDevice;
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

            var source = CreateTrainMinibatchSource(channelNameToMapFilePath, ctfFilePath, outputShape, maxSweeps);
            var targetStreamInfoName = source.StreamInfo(TargetsName);
            var sweeps = 0;

            while (true)
            {
                var minibatchData = source.GetNextMinibatch(minibatchSize, d);
                var targets = minibatchData[targetStreamInfoName];

                if(targets.sweepEnd)
                {
                    if (sweeps % 1000 == 0)
                    {
                        System.Console.WriteLine("Current sweep: " + sweeps);
                    }
                    sweeps++;
                }

                // Stop training once max epochs is reached.
                if (minibatchData.empty())
                {
                    System.Console.WriteLine($"Completed all {sweeps} sweeps");
                    break;
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
    }
}
