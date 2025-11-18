using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;

namespace ALPR;

public record AlprResult(DetectionResult Detection, OcrResult? Ocr);

public class AlprEngine(IDetector detector, IOcr ocr)
{
    public List<AlprResult> Predict(Image<Rgba32> frame)
    {
        var plateDetections = detector.Predict(frame);
        var results = new List<AlprResult>();

        foreach (var detection in plateDetections)
        {
            var bbox = detection.BoundingBox;

            var x1 = Math.Max(bbox.X1, 0);
            var y1 = Math.Max(bbox.Y1, 0);
            var x2 = Math.Min(bbox.X2, frame.Width);
            var y2 = Math.Min(bbox.Y2, frame.Height);
            
            var width = x2 - x1;
            var height = y2 - y1;

            if (width <= 0 || height <= 0) continue;
            using var croppedPlate = frame.Clone(ctx => ctx.Crop(new Rectangle(x1, y1, width, height)));
            var ocrResult = ocr.Predict(croppedPlate);

            results.Add(new AlprResult(detection, ocrResult));
        }

        return results;
    }

    public Image<Rgba32> DrawPredictions(Image<Rgba32> frame, List<AlprResult> results, SixLabors.Fonts.Font font)
    {
        var outputImage = frame.Clone();

        outputImage.Mutate(ctx =>
        {
            foreach (var result in results)
            {
                var bbox = result.Detection.BoundingBox;
                var rect = new Rectangle(bbox.X1, bbox.Y1, bbox.X2 - bbox.X1, bbox.Y2 - bbox.Y1);

                ctx.Draw(SixLabors.ImageSharp.Color.LimeGreen, 4, rect);

                if (result.Ocr == null || string.IsNullOrEmpty(result.Ocr.Text)) 
                    continue;

                var confidence = result.Ocr.Confidence.Any() 
                    ? result.Ocr.Confidence.Average() 
                    : 0.0f;

                var displayText = $"{result.Ocr.Text} {confidence:P0}";

                var location = new SixLabors.ImageSharp.PointF(bbox.X1, bbox.Y1 - 25);
                
                ctx.DrawText(displayText, font, Brushes.Solid(SixLabors.ImageSharp.Color.Black), new SixLabors.ImageSharp.PointF(location.X + 2, location.Y + 2));
                
                ctx.DrawText(displayText, font, Brushes.Solid(SixLabors.ImageSharp.Color.White), location);
            }
        });

        return outputImage;
    }
}