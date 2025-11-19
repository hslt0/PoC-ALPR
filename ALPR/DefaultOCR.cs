using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace ALPR;

/// <summary>
/// Default implementation of the Optical Character Recognition (OCR) engine.
/// This class is designed to work with "fast-plate-ocr" ONNX models (e.g., MobileViT-v2).
/// It handles image resizing, grayscale conversion, and decoding of model predictions into text.
/// </summary>
/// <param name="modelPath">The file path to the .onnx OCR model.</param>
/// <param name="alphabet">The string of characters the model was trained to recognize. Defaults to alphanumeric + underscore.</param>
/// <param name="height">The target image height required by the model input (default 70).</param>
/// <param name="width">The target image width required by the model input (default 140).</param>
/// <param name="maxSlots">The maximum number of characters (slots) the model predicts (default 8).</param>
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

    // Standard alphabet for global models usually includes digits, uppercase letters, and the underscore as padding.
    private const string DefaultAlphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";

    /// <summary>
    /// Performs text recognition on a cropped license plate image.
    /// </summary>
    /// <param name="croppedPlate">The cropped image containing only the license plate.</param>
    /// <returns>An <see cref="OcrResult"/> with the recognized text string and confidence scores.</returns>
    public OcrResult Predict(Image<Rgba32> croppedPlate)
    {
        // 1. Preprocess: Resize, Grayscale, and Convert to Tensor
        var inputTensor = Preprocess(croppedPlate);

        // 2. Setup input for ONNX
        var inputName = _session.InputMetadata.Keys.First();
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };

        // 3. Run Inference
        using var output = _session.Run(inputs);
        
        // 4. Get output tensor (Raw scores/logits)
        var outputTensor = output.First().AsTensor<float>();

        // 5. Decode output to string
        return DecodeOutput(outputTensor);
    }

    /// <summary>
    /// Prepares the image for the OCR model.
    /// </summary>
    /// <remarks>
    /// Steps:
    /// 1. Resizes the image to the specific dimensions (e.g., 140x70).
    /// 2. Converts the image to grayscale.
    /// 3. Creates a Byte Tensor in [Batch=1, Height, Width, Channels=1] format.
    /// Note: This specific model expects UINT8 input, not Float.
    /// </remarks>
    /// <param name="plate">The source image.</param>
    /// <returns>A DenseTensor of bytes ready for inference.</returns>
    private DenseTensor<byte> Preprocess(Image<Rgba32> plate)
    {
        using var processedImg = plate.Clone(x => 
        {
            x.Resize(width, height);
            x.Grayscale(); 
        });

        // Create tensor with shape [1, H, W, 1]
        var tensor = new DenseTensor<byte>(new[] { 1, height, width, 1 });

        processedImg.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < accessor.Height; y++)
            {
                var pixelRow = accessor.GetRowSpan(y);
                for (var x = 0; x < pixelRow.Length; x++)
                {
                    // Since image is grayscale, R=G=B. We take R channel as the byte value.
                    tensor[0, y, x, 0] = pixelRow[x].R; 
                }
            }
        });

        return tensor;
    }

    /// <summary>
    /// Decodes the raw model output into a readable string using ArgMax logic.
    /// </summary>
    /// <remarks>
    /// The model outputs a set of probabilities for every character in the alphabet for every "slot" (character position).
    /// This method finds the character with the highest probability for each slot.
    /// It skips characters that match the padding symbol ('_').
    /// </remarks>
    /// <param name="output">The raw float output from the model.</param>
    /// <returns>The decoded result.</returns>
    private OcrResult DecodeOutput(Tensor<float> output)
    {
        var vocabSize = _alphabet.Length;
        var slots = maxSlots;
        
        var data = output.ToDenseTensor().Buffer.Span;
        
        var resultText = "";
        var confidences = new List<float>();

        // Iterate through each character slot (e.g., 8 slots)
        for (var i = 0; i < slots; i++)
        {
            var maxVal = -float.MaxValue;
            var maxIdx = 0;

            // Find the character index with the highest score (ArgMax)
            for (var j = 0; j < vocabSize; j++)
            {
                // Calculate flat index: current_slot * alphabet_size + current_char_index
                var flatIndex = i * vocabSize + j;
                
                if (flatIndex >= data.Length) break;

                var val = data[flatIndex];
                
                if (val > maxVal)
                {
                    maxVal = val;
                    maxIdx = j;
                }
            }

            // Safety check
            if (maxIdx >= _alphabet.Length) continue;
            
            var predictedChar = _alphabet[maxIdx];
            
            // Ignore the padding character (usually '_')
            if (predictedChar == '_') continue;
            
            resultText += predictedChar;
            confidences.Add(maxVal);
        }

        return new OcrResult(resultText, confidences);
    }

    /// <summary>
    /// Disposes the ONNX InferenceSession.
    /// </summary>
    public void Dispose()
    {
        _session.Dispose();
        GC.SuppressFinalize(this);
    }
}