namespace WPF_FruitImageClassification
{
    /// <summary>
    /// Represents inputdata to train and evaluate the model
    /// </summary>
    public class ImageInput
    {
        /// <summary>
        /// Fuld sti til billedfilen.
        /// </summary>
        public string ImagePath { get; set; } = string.Empty;

        /// <summary>
        /// Klassen/label for billedet, fx "jordbær1", "æble" eller "pære".
        /// </summary>
        public string Label { get; set; } = string.Empty;
    }
}
