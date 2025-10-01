using Microsoft.ML;
using WPF_FruitImageClassification.Models;
using Xunit;

namespace FruitImageClassification_Tests
{
    /// <summary>
    /// Contains tests to verify that data can be loaded and that the model
    /// achieves a minimum accuracy when trained and evaluated.
    /// </summary>
    public class ImageTests
    {
        /// <summary>
        /// Ensures that the training folder contains at least one image file.
        /// </summary>
        [Fact]
        public void LoadTrainingData_ShouldReturnImages()
        {
            // Arrange: build absolute path to Train folder (copied to output directory)
            string trainPath = Path.Combine(AppContext.BaseDirectory, "Data", "Train");

            // Act: collect all image files from subfolders
            var imageFiles = Directory.GetDirectories(trainPath)
                                      .SelectMany(dir => Directory.GetFiles(dir));

            // Assert: fail if no images are found
            Assert.True(imageFiles.Any(),
                $"No training images found in: {trainPath}. " +
                "Check that your Train folder is marked 'Copy to Output Directory: Copy Always'.");
        }

        /// <summary>
        /// Trains the model and checks that accuracy is at least 70% on test data.
        /// </summary>
        [Fact]
        public void TrainedModel_ShouldReachMinimumAccuracy()
        {
            // Arrange: find absolute paths for Train and Test folders
            string trainPath = Path.Combine(AppContext.BaseDirectory, "Data", "Train");
            string testPath = Path.Combine(AppContext.BaseDirectory, "Data", "Test");

            var mlContext = new MLContext(seed: 1);

            // Load training data into IDataView
            var trainData = mlContext.Data.LoadFromEnumerable(
                Directory.GetDirectories(trainPath)
                    .SelectMany(dir => Directory.GetFiles(dir))
                    .Select(file => new ImageInput
                    {
                        ImagePath = file,
                        Label = Path.GetFileName(Path.GetDirectoryName(file)) ?? string.Empty
                    })
            );

            // Load test data into IDataView
            var testData = mlContext.Data.LoadFromEnumerable(
                Directory.GetDirectories(testPath)
                    .SelectMany(dir => Directory.GetFiles(dir))
                    .Select(file => new ImageInput
                    {
                        ImagePath = file,
                        Label = Path.GetFileName(Path.GetDirectoryName(file)) ?? string.Empty
                    })
            );

            // Build ML pipeline: encode labels, load images, train classifier, decode predicted labels
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.LoadRawImageBytes("Image", trainPath, "ImagePath"))
                .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(
                    labelColumnName: "Label", featureColumnName: "Image"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Act: train and evaluate
            var model = pipeline.Fit(trainData);
            var predictions = model.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            // Assert: ensure minimum accuracy
            Assert.True(metrics.MacroAccuracy > 0.70,
                $"Model accuracy too low: {metrics.MacroAccuracy:P2}. Add more images or improve data.");
        }
    }
}
