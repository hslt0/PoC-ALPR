using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ALPR;

record BoundingBox(int X1, int Y1, int X2, int Y2)
{
    public Rectangle ToRectangle() => new Rectangle(X1, Y1, X2 - X1, Y2 - Y1);
}

record DetectionResult(string Label, float Confidence, BoundingBox BoundingBox);

record OcrResult(string Text, List<float> Confidence);

interface IDetector
{
    public List<DetectionResult> Predict(Image<Rgba32> frame);
}

interface IOcr
{
    public OcrResult? Predict(Image<Rgba32> croppedPlate);
}