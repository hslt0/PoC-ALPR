using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;

namespace ALPR;

public class DefaultDetector(string modelPath, float confidenceThreshold = 0.4f) : IDetector, IDisposable
{
    private readonly InferenceSession _session = new(modelPath);
    private const int TargetSize = 384;

    public List<DetectionResult> Predict(Image<Rgba32> frame)
    {
        var (inputTensor, ratio, padding) = Preprocess(frame);

        var inputName = _session.InputMetadata.Keys.First();
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };

        using var output = _session.Run(inputs);
        var outputTensor = output[0].AsTensor<float>();

        return ParseOutput(outputTensor, ratio, padding);
    }

    private (DenseTensor<float> Tensor, float Ratio, (float dw, float dh) Padding) Preprocess(Image<Rgba32> originalImage)
    {
        var targetW = TargetSize;
        var targetH = TargetSize;

        var r = Math.Min((float)targetW / originalImage.Width, (float)targetH / originalImage.Height);
        
        var newUnpadW = (int)Math.Round(originalImage.Width * r);
        var newUnpadH = (int)Math.Round(originalImage.Height * r);

        var dw = (targetW - newUnpadW) / 2f;
        var dh = (targetH - newUnpadH) / 2f;

        using var canvas = new Image<Rgba32>(targetW, targetH);
        canvas.Mutate(x => x.Fill(SixLabors.ImageSharp.Color.FromRgb(114, 114, 114)));

        using var resizedImg = originalImage.Clone(x => x.Resize(newUnpadW, newUnpadH));

        var left = (int)Math.Round(dw - 0.1f);
        var top = (int)Math.Round(dh - 0.1f);
        
        // ReSharper disable once AccessToDisposedClosure
        canvas.Mutate(x => x.DrawImage(resizedImg, new SixLabors.ImageSharp.Point(left, top), 1f));

        var tensor = new DenseTensor<float>(new[] { 1, 3, targetH, targetW });

        canvas.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < accessor.Height; y++)
            {
                var pixelRow = accessor.GetRowSpan(y);
                for (var x = 0; x < pixelRow.Length; x++)
                {
                    var pixel = pixelRow[x];
                    tensor[0, 0, y, x] = pixel.R / 255.0f;
                    tensor[0, 1, y, x] = pixel.G / 255.0f;
                    tensor[0, 2, y, x] = pixel.B / 255.0f;
                }
            }
        });

        return (tensor, r, (dw, dh));
    }

    private List<DetectionResult> ParseOutput(Tensor<float> output, float ratio, (float dw, float dh) padding)
    {
        var results = new List<DetectionResult>();
        
        var floatCount = output.Length;
        var itemsCount = floatCount / 7;
        
        var data = output.ToDenseTensor().Buffer.Span; 

        for (var i = 0; i < itemsCount; i++)
        {
            var offset = i * 7;
            
            var score = data[offset + 6];
            if (score < confidenceThreshold)
                continue;
            
            var x1 = data[offset + 1];
            var y1 = data[offset + 2];
            var x2 = data[offset + 3];
            var y2 = data[offset + 4];
            
            var originalX1 = (x1 - padding.dw) / ratio;
            var originalY1 = (y1 - padding.dh) / ratio;
            var originalX2 = (x2 - padding.dw) / ratio;
            var originalY2 = (y2 - padding.dh) / ratio;

            var bbox = new BoundingBox(
                (int)originalX1, 
                (int)originalY1, 
                (int)originalX2, 
                (int)originalY2
            );

            results.Add(new DetectionResult("License Plate", score, bbox));
        }

        return results;
    }

    public void Dispose()
    {
        _session.Dispose();
        GC.SuppressFinalize(this);
    }
}