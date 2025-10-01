using Microsoft.ML;
using Microsoft.ML.Data;
using WPF_FruitImageClassification.Models;
namespace WPF_FruitImageClassification
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Find project root by going two folders up from bin\Debug\net8.0
            string project_Root = Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, @"..\..\.."));
            // Define path to save the trained model
            string model_Path = Path.Combine(project_Root, @"Models\fruitModel.zip");
            // Create Models directory if it doesn't exist
            if (!Directory.Exists(model_Path))
            {
                Directory.CreateDirectory(project_Root + @"\Models");
            }
            // INITIALISERING
            // MLContext is a gateway til ML.NET and makes it easy to reproduce results
            var mlContext = new MLContext(seed: 1);

            // === DATA ===

            // Build paths to training and test folders from project root
            string trainPath = Path.Combine(project_Root, "Data", "Train");
            string testPath = Path.Combine(project_Root, "Data", "Test");

            // Load all training data:
            // 1. Get all subfolders under the training path
            // 2. Get all files in each folder
            // 3. Create an ImageInput object for each image with its path and label (folder name)
            var trainData = mlContext.Data.LoadFromEnumerable(
                Directory.GetDirectories(trainPath)
                        .SelectMany(dir => Directory.GetFiles(dir))
                        .Select(file => new ImageInput
                        {
                            ImagePath = file,                                      // Full path to the image file
                            Label = Path.GetFileName(Path.GetDirectoryName(file)) // Folder name becomes the label
                        })
            );

            // Load all test data in the same way, so we can evaluate the model later
            var testData = mlContext.Data.LoadFromEnumerable(
                Directory.GetDirectories(testPath)
                        .SelectMany(dir => Directory.GetFiles(dir))
                        .Select(file => new ImageInput
                        {
                            ImagePath = file,                                      // Full path to the image file
                            Label = Path.GetFileName(Path.GetDirectoryName(file)) // Folder name becomes the label
                        })
            );

            // === PIPELINE ===
            // Defines the data processing and training steps.
            var pipeline = mlContext.Transforms.Conversion
                // Convert string labels (e.g. "æble") to numeric keys for training
                .MapValueToKey("Label", "Label")

                // Load images as byte arrays from the train folder using ImagePath column
                .Append(mlContext.Transforms.LoadRawImageBytes(
                    outputColumnName: "Image",
                    imageFolder: trainPath,
                    inputColumnName: "ImagePath"))

                // Train image classification model using transfer learning
                .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(
                    labelColumnName: "Label",
                    featureColumnName: "Image"))

                // Convert predicted keys back to original label names
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // === TRAINING ===
            // Train the model using the pipeline and training data
            Console.WriteLine("Training model...");
            var model = pipeline.Fit(trainData);

            // Save the trained model to a .zip file so it can be reused later
            mlContext.Model.Save(model, trainData.Schema, model_Path);
            Console.WriteLine($"Model saved to: {model_Path}");

            // === EVALUATION ===

            // Use the trained model to make predictions on the test dataset
            var predictions = model.Transform(testData);

            // Calculate classification metrics based on the predictions and true labels
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            // Print key metrics to the console for performance review
            Console.WriteLine($"Accuracy: {metrics.MacroAccuracy:P2}");   // Overall accuracy (0–100%)
            Console.WriteLine($"LogLoss: {metrics.LogLoss:F4}");          // How confident the model is in predictions
            Console.WriteLine("Confusion Matrix:");                       // Shows correct vs. incorrect predictions per class
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

            WriteBenchmarkResults(project_Root, "fruitModel", metrics);
        }

        private static void WriteBenchmarkResults(string project_Root, string modelName, MulticlassClassificationMetrics metrics)
        {
            string sharedRessources = Path.Combine(project_Root, "SharedRessources");
            if (!Directory.Exists(sharedRessources))
            {
                Directory.CreateDirectory(sharedRessources);
            }
            string results = $"{modelName}\n" +
                             $"F1: {metrics.LogLoss:F1}\n" +
                             $"Accuracy: {metrics.MacroAccuracy:P2}\n" +
                             $"{metrics.ConfusionMatrix.GetFormattedConfusionTable()}";
            string resultsPath = Path.Combine(sharedRessources, "BenchmarkResults.csv");
            File.AppendAllText(resultsPath, results + Environment.NewLine);
            Console.WriteLine($"Benchmark results saved to: {resultsPath}");
        }
    }
}
