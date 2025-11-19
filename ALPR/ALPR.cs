using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;

namespace ALPR;

/// <summary>
/// Represents a single ALPR result, containing both the detection data (bounding box)
/// and the recognized text (OCR) for a specific license plate.
/// </summary>
public record AlprResult(DetectionResult Detection, OcrResult? Ocr);

/// <summary>
/// The main orchestration engine for the Automatic License Plate Recognition system.
/// It coordinates the workflow between the object detector and the OCR model.
/// </summary>
public class AlprEngine(IDetector detector, IOcr ocr)
{
    /// <summary>
    /// Executes the full ALPR pipeline on the provided image frame.
    /// </summary>
    /// <remarks>
    /// The pipeline consists of the following steps:
    /// 1. Detecting license plate bounding boxes using the detector.
    /// 2. Cropping the image to the detected regions of interest.
    /// 3. Running the OCR model on each cropped region to extract text.
    /// </remarks>
    /// <param name="frame">The input image to be processed.</param>
    /// <returns>A list of results containing detections and their corresponding recognized text.</returns>
    public List<AlprResult> Predict(Image<Rgba32> frame)
    {
        var plateDetections = detector.Predict(frame);
        var results = new List<AlprResult>();

        foreach (var detection in plateDetections)
        {
            var bbox = detection.BoundingBox;

            // Ensure cropping coordinates are within the image bounds
            var x1 = Math.Max(bbox.X1, 0);
            var y1 = Math.Max(bbox.Y1, 0);
            var x2 = Math.Min(bbox.X2, frame.Width);
            var y2 = Math.Min(bbox.Y2, frame.Height);
            
            var width = x2 - x1;
            var height = y2 - y1;

            if (width <= 0 || height <= 0) continue;

            // Clone and crop the specific region for OCR processing
            using var croppedPlate = frame.Clone(ctx => ctx.Crop(new Rectangle(x1, y1, width, height)));
            var ocrResult = ocr.Predict(croppedPlate);

            results.Add(new AlprResult(detection, ocrResult));
        }

        return results;
    }
    
    /// <summary>
    /// Visualizes the ALPR results by drawing bounding boxes and recognized text 
    /// directly onto a copy of the input image.
    /// </summary>
    /// <param name="frame">The source image to draw on.</param>
    /// <param name="results">The list of ALPR results to visualize.</param>
    /// <param name="font">The font used for rendering the license plate text.</param>
    /// <returns>A new image instance with the visualizations applied.</returns>
    // Method never used but original Python library contains it
    public Image<Rgba32> DrawPredictions(Image<Rgba32> frame, List<AlprResult> results, SixLabors.Fonts.Font font)
    {
        var outputImage = frame.Clone();

        outputImage.Mutate(ctx =>
        {
            foreach (var result in results)
            {
                var bbox = result.Detection.BoundingBox;
                var rect = new Rectangle(bbox.X1, bbox.Y1, bbox.X2 - bbox.X1, bbox.Y2 - bbox.Y1);

                // Draw bounding box
                ctx.Draw(SixLabors.ImageSharp.Color.LimeGreen, 4, rect);

                if (result.Ocr == null || string.IsNullOrEmpty(result.Ocr.Text)) 
                    continue;

                var confidence = result.Ocr.Confidence.Any() 
                    ? result.Ocr.Confidence.Average() 
                    : 0.0f;

                var displayText = $"{result.Ocr.Text} {confidence:P0}";

                var location = new SixLabors.ImageSharp.PointF(bbox.X1, bbox.Y1 - 25);
                
                // Draw text with a black outline (shadow) for better visibility
                ctx.DrawText(displayText, font, Brushes.Solid(SixLabors.ImageSharp.Color.Black), new SixLabors.ImageSharp.PointF(location.X + 2, location.Y + 2));
                
                // Draw main white text
                ctx.DrawText(displayText, font, Brushes.Solid(SixLabors.ImageSharp.Color.White), location);
            }
        });

        return outputImage;
    }
}