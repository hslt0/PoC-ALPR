using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ALPR;

/// <summary>
/// Represents a rectangular bounding box defined by its top-left (X1, Y1) and bottom-right (X2, Y2) coordinates.
/// </summary>
/// <param name="X1">The X-coordinate of the top-left corner.</param>
/// <param name="Y1">The Y-coordinate of the top-left corner.</param>
/// <param name="X2">The X-coordinate of the bottom-right corner.</param>
/// <param name="Y2">The Y-coordinate of the bottom-right corner.</param>
public record BoundingBox(int X1, int Y1, int X2, int Y2);

/// <summary>
/// Represents a single object detection result found by the detector model.
/// </summary>
/// <param name="Label">The class label of the detected object (e.g., "License Plate").</param>
/// <param name="Confidence">The confidence score of the detection (typically between 0.0 and 1.0).</param>
/// <param name="BoundingBox">The coordinates of the detected object.</param>
public record DetectionResult(string Label, float Confidence, BoundingBox BoundingBox);

/// <summary>
/// Represents the output of the OCR process for a specific image region.
/// </summary>
/// <param name="Text">The recognized text string.</param>
/// <param name="Confidence">A list of confidence scores corresponding to each recognized character.</param>
public record OcrResult(string Text, List<float> Confidence);

/// <summary>
/// Defines the contract for an object detector capable of locating license plates within an image.
/// </summary>
public interface IDetector
{
    /// <summary>
    /// Performs object detection on the provided image frame.
    /// </summary>
    /// <param name="frame">The input image to be analyzed.</param>
    /// <returns>A list of detected objects with their bounding boxes and confidence scores.</returns>
    public List<DetectionResult> Predict(Image<Rgba32> frame);
}

/// <summary>
/// Defines the contract for an Optical Character Recognition (OCR) model capable of reading text from an image.
/// </summary>
public interface IOcr
{
    /// <summary>
    /// Performs text recognition on a cropped image of a license plate.
    /// </summary>
    /// <param name="croppedPlate">The cropped image containing only the license plate.</param>
    /// <returns>An <see cref="OcrResult"/> containing the text and confidence, or null if no text could be recognized.</returns>
    public OcrResult? Predict(Image<Rgba32> croppedPlate);
}