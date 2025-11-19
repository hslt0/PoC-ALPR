using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ALPR;

public record BoundingBox(int X1, int Y1, int X2, int Y2);

public record DetectionResult(string Label, float Confidence, BoundingBox BoundingBox);

public record OcrResult(string Text, List<float> Confidence);

public interface IDetector
{
    public List<DetectionResult> Predict(Image<Rgba32> frame);
}

public interface IOcr
{
    public OcrResult? Predict(Image<Rgba32> croppedPlate);
}