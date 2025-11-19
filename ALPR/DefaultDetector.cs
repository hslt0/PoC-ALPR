using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;

namespace ALPR;

/// <summary>
/// Default implementation of the object detector using YOLOv9 ONNX models.
/// This class handles the specific preprocessing (letterboxing) and postprocessing 
/// required for YOLO "End-to-End" models.
/// </summary>
/// <param name="modelPath">The file path to the .onnx model.</param>
/// <param name="confidenceThreshold">The minimum confidence score (0.0 - 1.0) required to accept a detection. Defaults to 0.4.</param>
public class DefaultDetector(string modelPath, float confidenceThreshold = 0.4f) : IDetector, IDisposable
{
    private readonly InferenceSession _session = new(modelPath);
    
    // YOLOv9 specific input size. The model expects 384x384 images.
    private const int TargetSize = 384;

    /// <summary>
    /// Runs object detection on the provided image frame.
    /// </summary>
    /// <param name="frame">The input image to analyze.</param>
    /// <returns>A list of detected license plates with their bounding boxes and confidence scores.</returns>
    public List<DetectionResult> Predict(Image<Rgba32> frame)
    {
        // 1. Preprocess the image to match model input requirements
        var (inputTensor, ratio, padding) = Preprocess(frame);

        // 2. Prepare input for ONNX Runtime
        var inputName = _session.InputMetadata.Keys.First();
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };

        // 3. Run inference
        using var output = _session.Run(inputs);
        
        // 4. Get the first output tensor (usually named "output0" or similar)
        var outputTensor = output[0].AsTensor<float>();

        // 5. Parse raw float data into detection results
        return ParseOutput(outputTensor, ratio, padding);
    }

    /// <summary>
    /// Prepares the image for the YOLO model using "Letterbox" resizing.
    /// </summary>
    /// <remarks>
    /// YOLO models expect square inputs (e.g., 384x384). To avoid distorting the image aspect ratio:
    /// 1. The image is scaled down to fit within the target size.
    /// 2. The remaining space is padded with a gray color (114, 114, 114).
    /// 3. Pixel values are normalized to [0.0, 1.0].
    /// </remarks>
    /// <param name="originalImage">The source image.</param>
    /// <returns>
    /// A tuple containing:
    /// <br/>- <b>Tensor:</b> The processed image data in NCHW format.
    /// <br/>- <b>Ratio:</b> The scaling factor used.
    /// <br/>- <b>Padding:</b> The (width, height) padding added to center the image.
    /// </returns>
    private (DenseTensor<float> Tensor, float Ratio, (float dw, float dh) Padding) Preprocess(Image<Rgba32> originalImage)
    {
        var targetW = TargetSize;
        var targetH = TargetSize;

        // Calculate scaling ratio (min of width/height ratios) to fit inside target box
        var r = Math.Min((float)targetW / originalImage.Width, (float)targetH / originalImage.Height);
        
        // Calculate new dimensions
        var newUnpadW = (int)Math.Round(originalImage.Width * r);
        var newUnpadH = (int)Math.Round(originalImage.Height * r);

        // Calculate padding needed to fill the square
        var dw = (targetW - newUnpadW) / 2f;
        var dh = (targetH - newUnpadH) / 2f;

        // Create a canvas filled with gray (YOLO standard padding color)
        using var canvas = new Image<Rgba32>(targetW, targetH);
        canvas.Mutate(x => x.Fill(SixLabors.ImageSharp.Color.FromRgb(114, 114, 114)));

        // Resize original image
        using var resizedImg = originalImage.Clone(x => x.Resize(newUnpadW, newUnpadH));

        var left = (int)Math.Round(dw - 0.1f);
        var top = (int)Math.Round(dh - 0.1f);
        
        // Draw resized image onto the center of the gray canvas
        // ReSharper disable once AccessToDisposedClosure
        canvas.Mutate(x => x.DrawImage(resizedImg, new SixLabors.ImageSharp.Point(left, top), 1f));

        // Create tensor with shape [Batch=1, Channels=3, Height, Width]
        var tensor = new DenseTensor<float>(new[] { 1, 3, targetH, targetW });

        // Copy pixels to tensor and normalize (0-255 -> 0.0-1.0)
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

    /// <summary>
    /// Parses the raw output tensor from the YOLO model into structured results.
    /// </summary>
    /// <remarks>
    /// The End-to-End model output is a flat array where every 7 floats represent one detection box:
    /// [batch_index, x1, y1, x2, y2, class_id, confidence].
    /// This method filters by confidence and maps the resized/padded coordinates back 
    /// to the original image coordinates.
    /// </remarks>
    /// <param name="output">The raw output tensor from ONNX Runtime.</param>
    /// <param name="ratio">The scaling ratio used in preprocessing.</param>
    /// <param name="padding">The padding values used in preprocessing.</param>
    /// <returns>A list of <see cref="DetectionResult"/> objects.</returns>
    private List<DetectionResult> ParseOutput(Tensor<float> output, float ratio, (float dw, float dh) padding)
    {
        var results = new List<DetectionResult>();
        
        var floatCount = output.Length;
        var itemsCount = floatCount / 7; // Each box has 7 values
        
        var data = output.ToDenseTensor().Buffer.Span; 

        for (var i = 0; i < itemsCount; i++)
        {
            var offset = i * 7;
            
            var score = data[offset + 6];
            if (score < confidenceThreshold)
                continue;
            
            // Extract coordinates (relative to the 384x384 canvas)
            var x1 = data[offset + 1];
            var y1 = data[offset + 2];
            var x2 = data[offset + 3];
            var y2 = data[offset + 4];
            
            // Map back to original image coordinates by removing padding and scaling up
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

    /// <summary>
    /// Disposes the ONNX InferenceSession to release unmanaged resources.
    /// </summary>
    public void Dispose()
    {
        _session.Dispose();
        GC.SuppressFinalize(this);
    }
}