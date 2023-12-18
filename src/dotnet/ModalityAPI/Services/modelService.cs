using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.Linq;
using System;

public class ModelService
{
    private readonly Dictionary<string, string> _modelPaths;
    private readonly Dictionary<string, Lazy<InferenceSession>> _sessions;

    public ModelService(IConfiguration configuration)
    {
        _modelPaths = configuration.GetSection("ModelPaths").Get<Dictionary<string, string>>() ?? throw new InvalidOperationException("No model directories found") ;

        if (!_modelPaths.Any())
        {
            throw new InvalidOperationException("Model paths are not configured.");
        }
        // Init model inference sessions and nest within a dictionary
        _sessions = _modelPaths.ToDictionary(
            pair => pair.Key,
            pair => new Lazy<InferenceSession>(() => new InferenceSession(pair.Value))
        );
    }
    public string GetModelPath(string modelName)
    {
        if (_modelPaths.TryGetValue(modelName, out var modelPath))
        {
            return modelPath;
        }
        else
        {
            throw new ArgumentException($"Model name '{modelName}' is not recognized.");
        }
    }
    public List<string> GetInputNames(string modelName)
    {
        if (!_sessions.TryGetValue(modelName, out var lazySession))
        {
            throw new ArgumentException($"Model name '{modelName}' is not recognized.");
        }

        return lazySession.Value.InputMetadata.Keys.ToList();
    }
    public List<string> GetOutputNames(string modelName)
    {
        if (!_sessions.TryGetValue(modelName, out var lazySession))
        {
            throw new ArgumentException($"Model name '{modelName}' is not recognized.");
        }

        return lazySession.Value.OutputMetadata.Keys.ToList();
    }
    public int GetInputDimension(string modelName)
    {
        if (!_sessions.TryGetValue(modelName, out var lazySession))
        {
            throw new ArgumentException($"Model name '{modelName}' is not recognized.");
        }

        var inputMetaData = lazySession.Value.InputMetadata;
        var firstInputName = inputMetaData.Keys.First();
        var inputShape = inputMetaData[firstInputName].Dimensions;

        // Assuming the input dimension you're interested in is the last one
        return inputShape.Last(); 
    }
    public List<float[]> RunModel(string modelName, List<float> inputVector)
    {
        if (!_sessions.TryGetValue(modelName, out var lazySession))
        {
            throw new ArgumentException($"Model name '{modelName}' is not recognized.");
        }

        var session = lazySession.Value;

        var tensor = new DenseTensor<float>(inputVector.ToArray(), new[] { 1, 1, inputVector.Count });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(session.InputMetadata.Keys.First(), tensor)
        };

        using var results = session.Run(inputs);

        // Process all outputs
        var outputVectors = new List<float[]>();
        foreach (var name in session.OutputMetadata.Keys)
        {
            var outputTensor = results.FirstOrDefault(item => item.Name == name)?.AsTensor<float>();
            if (outputTensor != null)
            {
                outputVectors.Add(outputTensor.ToArray());
            }
        }

        return outputVectors;
    }
}
