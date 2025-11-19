using ALPR;
using SixLabors.ImageSharp.PixelFormats;
using Image = SixLabors.ImageSharp.Image;

namespace App.Services;

public class AlprService
{
    private AlprEngine _engine = null!;

    public bool IsInitialized { get; private set; }

    public async Task InitializeAsync()
    {
        if (IsInitialized) return;

        try
        {
            var yoloPath = await CopyAssetToCacheAsync("yolo.onnx");
            var ocrPath = await CopyAssetToCacheAsync("ocr.onnx");

            var detector = new DefaultDetector(yoloPath);
            var ocr = new DefaultOcr(ocrPath);
            _engine = new AlprEngine(detector, ocr);
            
            IsInitialized = true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"ALPR Init Error: {ex.Message}");
            throw;
        }
    }

    public string ProcessImage(Stream imageStream)
    {
        if (!IsInitialized) return "System not initialized";

        try
        {
            if (imageStream.Position != 0) imageStream.Position = 0;

            using var image = Image.Load<Rgba32>(imageStream);
            var results = _engine.Predict(image);

            if (results.Count == 0) return "No license plates found";

            var text = "";
            foreach (var res in results)
            {
                var plate = res.Ocr?.Text ?? "---";
                var conf = res.Ocr?.Confidence.Count > 0 
                    ? $"{res.Ocr.Confidence.Average():P0}" 
                    : "0%";
                text += $"{plate} ({conf})\n";
            }
            return text;
        }
        catch (Exception ex)
        {
            return $"Processing error: {ex.Message}";
        }
    }

    private async Task<string> CopyAssetToCacheAsync(string filename)
    {
        var targetFile = Path.Combine(FileSystem.CacheDirectory, filename);
        if (File.Exists(targetFile)) return targetFile;

        await using var inputStream = await FileSystem.Current.OpenAppPackageFileAsync(filename);
        await using var outputStream = File.Create(targetFile);
        await inputStream.CopyToAsync(outputStream);
        return targetFile;
    }
}