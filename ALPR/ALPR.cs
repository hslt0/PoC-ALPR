using SkiaSharp;

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
    /// <param name="frame">The input image to be processed (as SKBitmap).</param>
    /// <returns>A list of results containing detections and their corresponding recognized text.</returns>
    public List<AlprResult> Predict(SKBitmap frame) // Зміна типу входу
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
            var cropRect = new SKRectI(x1, y1, x2, y2);
            using var croppedImage = SKImage.FromBitmap(frame).Subset(cropRect);
            using var croppedPlate = SKBitmap.FromImage(croppedImage);
            
            var ocrResult = ocr.Predict(croppedPlate);

            results.Add(new AlprResult(detection, ocrResult));
        }

        return results;
    }
    
    /// <summary>
    /// Visualizes the ALPR results by drawing bounding boxes and recognized text 
    /// directly onto a copy of the input image.
    /// </summary>
    /// <remarks>
    /// NOTE: This drawing implementation uses a fallback font and requires 
    /// the calling code to handle the font size and color definitions via SKPaint.
    /// </remarks>
    /// <param name="frame">The source image to draw on (as SKBitmap).</param>
    /// <param name="results">The list of ALPR results to visualize.</param>
    /// <param name="skFont">The font used for rendering the license plate text. (NOTE: This object type is ignored; font is drawn using SkiaSharp's defaults for simplicity).</param>
    /// <returns>A new image instance with the visualizations applied (as SKBitmap).</returns>
   public SKBitmap DrawPredictions(SKBitmap frame, List<AlprResult> results, SKFont skFont) 
    {
        using var outputImage = frame.Copy(); 
        using var canvas = new SKCanvas(outputImage);
        
        using var strokePaint = new SKPaint();
        strokePaint.Color = SKColors.LimeGreen;
        strokePaint.Style = SKPaintStyle.Stroke;
        strokePaint.StrokeWidth = 4;
        strokePaint.IsAntialias = true;

        using var textPaintBlack = new SKPaint();
        textPaintBlack.Color = SKColors.Black;
        textPaintBlack.Style = SKPaintStyle.StrokeAndFill;
        textPaintBlack.StrokeWidth = 6;
        textPaintBlack.IsAntialias = true;

        using var textPaintWhite = new SKPaint();
        textPaintWhite.Color = SKColors.White;
        textPaintWhite.IsAntialias = true;

        foreach (var result in results)
        {
            var bbox = result.Detection.BoundingBox;
            var rect = new SKRect(bbox.X1, bbox.Y1, bbox.X2, bbox.Y2);
            
            canvas.DrawRect(rect, strokePaint);

            if (result.Ocr == null || string.IsNullOrEmpty(result.Ocr.Text)) 
                continue;

            var confidence = result.Ocr.Confidence.Any() 
                ? result.Ocr.Confidence.Average() 
                : 0.0f;

            var displayText = $"{result.Ocr.Text} {confidence:P0}";
            
            var location = new SKPoint(bbox.X1, bbox.Y1 - 10);
            
            canvas.DrawText(displayText, location, SKTextAlign.Left, skFont, textPaintBlack); 
            
            canvas.DrawText(displayText, location, SKTextAlign.Left, skFont, textPaintWhite); 
        }
        
        return outputImage.Copy(); 
    }
}