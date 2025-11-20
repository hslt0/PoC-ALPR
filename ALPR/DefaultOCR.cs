using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace ALPR;

/// <summary>
/// Default implementation of the Optical Character Recognition (OCR) engine.
/// Handles image preprocessing (resizing, RGB conversion) and runs inference using ONNX Runtime with SkiaSharp.
/// </summary>
/// <param name="modelPath">The file path to the .onnx OCR model.</param>
/// <param name="alphabet">The string of characters the model was trained to recognize (including padding).</param>
/// <param name="height">The target image height required by the model input.</param>
/// <param name="width">The target image width required by the model input.</param>
/// <param name="maxSlots">The maximum number of characters (slots) the model predicts.</param>
public class DefaultOcr(
    string modelPath,
    string alphabet = DefaultOcr.DefaultAlphabet,
    int height = 64,
    int width = 128, 
    int maxSlots = 9)
    : IOcr, IDisposable
{
    private readonly InferenceSession _session = new(modelPath);
    private readonly char[] _alphabet = alphabet.ToCharArray();

    /// <summary>
    /// Standard alphabet for global models, including alphanumeric characters and the underscore as a padding symbol.
    /// </summary>
    private const string DefaultAlphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";

    /// <summary>
    /// Performs text recognition on a cropped license plate image.
    /// </summary>
    /// <param name="croppedPlate">The cropped image containing only the license plate (as SKBitmap).</param>
    /// <returns>An <see cref="OcrResult"/> containing the recognized text and confidence scores.</returns>
    public OcrResult Predict(SKBitmap croppedPlate)
    {
        // 1. Preprocess the image into a tensor
        var inputTensor = Preprocess(croppedPlate);

        // 2. Prepare inputs for ONNX Runtime
        var inputName = _session.InputMetadata.Keys.First();
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };

        // 3. Run inference
        using var output = _session.Run(inputs);
        
        // 4. Extract the output tensor
        var outputTensor = output.First().AsTensor<float>();

        // 5. Decode the results into a string
        return DecodeOutput(outputTensor);
    }

    /// <summary>
    /// Converts the input image into a Byte Tensor (UINT8) required by the model.
    /// Resizes the image and maps pixels to an RGB format [1, Height, Width, 3].
    /// </summary>
    /// <param name="plate">The source image.</param>
    /// <returns>A dense tensor of bytes representing the image.</returns>
    /// <exception cref="InvalidOperationException">Thrown if pixel data cannot be accessed safely.</exception>
    private DenseTensor<byte> Preprocess(SKBitmap plate)
    {
        // Resize to target dimensions using standard sampling
        var info = new SKImageInfo(width, height, SKColorType.Rgba8888);
        using var resizedBitmap = plate.Resize(info, SKSamplingOptions.Default);

        // Create a tensor with shape [Batch=1, Height, Width, Channels=3]
        var tensor = new DenseTensor<byte>(new[] { 1, height, width, 3 });

        var pixmap = new SKPixmap();
        
        // Ensure we can access the pixel memory safely
        if (!resizedBitmap.PeekPixels(pixmap))
        {
            throw new InvalidOperationException("Could not get safe pixel access from resized SKBitmap.");
        }
        
        var pixelSpan = pixmap.GetPixelSpan<SKColor>(); 
        
        // Iterate through pixels and populate the tensor
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var pixelIndex = y * width + x; 
                var color = pixelSpan[pixelIndex];

                // Populate RGB channels
                tensor[0, y, x, 0] = color.Red;   // R
                tensor[0, y, x, 1] = color.Green; // G
                tensor[0, y, x, 2] = color.Blue;  // B
            }
        }

        return tensor;
    }

    /// <summary>
    /// Decodes the raw model output using ArgMax logic.
    /// Finds the character with the highest probability for each slot and constructs the result string.
    /// </summary>
    /// <param name="output">The raw float output tensor from the model.</param>
    /// <returns>The decoded OCR result.</returns>
    private OcrResult DecodeOutput(Tensor<float> output)
    {
        var slots = maxSlots;
        var vocabSize = alphabet.Length;

        var data = output.ToDenseTensor().Buffer.Span;
        
        var resultText = "";
        var confidences = new List<float>();

        // Iterate over each character slot
        for (var i = 0; i < slots; i++)
        {
            var maxVal = -float.MaxValue;
            var maxIdx = 0;

            // Find the index with the highest score in the vocabulary
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
            
            // Skip the padding character
            if (predictedChar == '_') continue;
            
            resultText += predictedChar;
            confidences.Add(maxVal);
        }

        return new OcrResult(resultText, confidences);
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