using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace ALPR;

public class DefaultOcr(
    string modelPath,
    string alphabet = DefaultOcr.DefaultAlphabet,
    int height = 70,
    int width = 140,
    int maxSlots = 8)
    : IOcr, IDisposable
{
    private readonly InferenceSession _session = new(modelPath);
    private readonly char[] _alphabet = alphabet.ToCharArray();

    private const string DefaultAlphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";

    public OcrResult Predict(Image<Rgba32> croppedPlate)
    {
        var inputTensor = Preprocess(croppedPlate);

        var inputName = _session.InputMetadata.Keys.First();
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };

        using var output = _session.Run(inputs);
        
        var outputTensor = output.First().AsTensor<float>();

        return DecodeOutput(outputTensor);
    }

    private DenseTensor<byte> Preprocess(Image<Rgba32> plate)
    {
        using var processedImg = plate.Clone(x => 
        {
            x.Resize(width, height);
            x.Grayscale(); 
        });

        var tensor = new DenseTensor<byte>(new[] { 1, height, width, 1 });

        processedImg.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < accessor.Height; y++)
            {
                var pixelRow = accessor.GetRowSpan(y);
                for (var x = 0; x < pixelRow.Length; x++)
                {
                    tensor[0, y, x, 0] = pixelRow[x].R; 
                }
            }
        });

        return tensor;
    }

    private OcrResult DecodeOutput(Tensor<float> output)
    {
        var vocabSize = _alphabet.Length;
        var slots = maxSlots;
        
        var data = output.ToDenseTensor().Buffer.Span;
        
        var resultText = "";
        var confidences = new List<float>();

        for (var i = 0; i < slots; i++)
        {
            var maxVal = -float.MaxValue;
            var maxIdx = 0;

            for (var j = 0; j < vocabSize; j++)
            {
                var flatIndex = i * vocabSize + j;
                
                if (flatIndex >= data.Length) break;

                var val = data[flatIndex];
                
                if (val > maxVal)
                {
                    maxVal = val;
                    maxIdx = j;
                }
            }

            if (maxIdx >= _alphabet.Length) continue;
            
            var predictedChar = _alphabet[maxIdx];
            
            if (predictedChar == '_') continue;
            
            resultText += predictedChar;
            confidences.Add(maxVal);
        }

        return new OcrResult(resultText, confidences);
    }

    public void Dispose()
    {
        _session.Dispose();
    }
}