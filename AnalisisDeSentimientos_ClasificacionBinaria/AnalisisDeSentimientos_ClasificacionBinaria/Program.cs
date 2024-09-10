#region Documentación
/*
 * Este programa su finalidad es mostrar cómo realizar análisis de sentimientos en texto mediante la clasificación binaria.
 * Se utiliza un modelo de regresión logística para clasificar el sentimiento de las opiniones de los clientes en positivo o negativo.
 * 
 * 1. El programa carga los datos de un archivo de texto y los divide en conjuntos de entrenamiento y prueba.
 * 2. Construye y entrena un modelo de análisis de sentimientos.
 * 3. Evalúa el modelo con el conjunto de datos de prueba.
 * 4. Realiza predicciones con el modelo.
 *
 * Referencias y documentación: https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/sentiment-analysis
 *
 */
#endregion

using Microsoft.ML;
using Microsoft.ML.Data;


using static Microsoft.ML.DataOperationsCatalog;

namespace AnalisisDeSentimientos_ClasificacionBinaria
{
    /// <summary>
    /// Clase principal del programa de análisis de sentimientos.
    /// </summary>
    public class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Datos", "yelp_labelled.txt");

        /// <summary>
        /// Punto de entrada principal para la aplicación.
        /// </summary>
        /// <param name="args">Argumentos de línea de comandos.</param>
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            // Llama al método LoadData para cargar y dividir los datos en conjuntos de entrenamiento y prueba.
            TrainTestData splitDataView = LoadData(mlContext);

            // Llama al método BuildAndTrainModel para construir y entrenar el modelo.
            ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);

            // Llama al método Evaluate para evaluar el modelo con el conjunto de datos de prueba.
            Evaluate(mlContext, model, splitDataView.TestSet);

            // Llama a los métodos UseModelWithSingleItem y UseModelWithBatchItems para realizar predicciones con el modelo.
            UseModelWithSingleItem(mlContext, model);

            UseModelWithBatchItems(mlContext, model);

            Console.WriteLine();
            Console.WriteLine("=============== Fin del proceso ===============");
        }

        /// <summary>
        /// Carga los datos desde el archivo de texto y los divide en conjuntos de entrenamiento y prueba.
        /// </summary>
        /// <param name="mlContext">El contexto de ML.</param>
        /// <returns>Los datos divididos en conjuntos de entrenamiento y prueba.</returns>
        public static TrainTestData LoadData(MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            return splitDataView;
        }

        /// <summary>
        /// Construye y entrena el modelo de análisis de sentimientos.
        /// </summary>
        /// <param name="mlContext">El contexto de ML.</param>
        /// <param name="splitTrainSet">El conjunto de datos de entrenamiento.</param>
        /// <returns>El modelo entrenado.</returns>
        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));
            // entrena con regresion 
            Console.WriteLine("=============== Crear y entrena el modelo ===============");
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== Fin de la formación ===============");
            Console.WriteLine();

            return model;
        }

        /// <summary>
        /// Evalúa el modelo con el conjunto de datos de prueba.
        /// </summary>
        /// <param name="mlContext">El contexto de ML.</param>
        /// <param name="model">El modelo entrenado.</param>
        /// <param name="splitTestSet">El conjunto de datos de prueba.</param>
        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.WriteLine("=============== Evaluación de la precisión del modelo con datos de prueba ===============");
            IDataView predictions = model.Transform(splitTestSet);
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("Evaluación de métricas de calidad del modelo");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}"); // Accuracy obtiene la precisión de un modelo, que corresponde a la proporción de predicciones correctas en el conjunto de pruebas.
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}"); //AreaUnderRocCurve indica el nivel de confianza del modelo en clasificar correctamente las clases positivas y negativas.
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");//F1Score es la media armónica de la precisión y la exhaustividad.
            Console.WriteLine("=============== Fin de la evaluación del modelo ===============");
        }

        /// <summary>
        /// Usa el modelo para predecir el sentimiento de una única muestra.
        /// </summary>
        /// <param name="mlContext">El contexto de ML.</param>
        /// <param name="model">El modelo entrenado.</param>
        private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "This was a very good steak." // - Este fue un bistec muy malo
            };

            var resultPrediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prueba de predicción del modelo con una única muestra y un único conjunto de datos de prueba ===============");
            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")}  ");
            Console.WriteLine("=============== Fin de las predicciones ===============");
            Console.WriteLine();
        }

        /// <summary>
        /// Usa el modelo para predecir el sentimiento de múltiples muestras.
        /// </summary>
        /// <param name="mlContext">El contexto de ML.</param>
        /// <param name="model">El modelo entrenado.</param>
        public static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
        {
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "This was a horrible meal " // - Esta fue una comida horrible
                },
                new SentimentData
                {
                    SentimentText = "I love this spaghetti." // - Me encanta este espagueti
                }
            };

            IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);
            IDataView predictions = model.Transform(batchComments);
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

            Console.WriteLine();
            Console.WriteLine("=============== Prueba de predicción de un modelo cargado con múltiples muestras ===============");
            Console.WriteLine();

            foreach (SentimentPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")}  ");
            }
            Console.WriteLine("=============== Fin de la predicción ===============");
        }
    }

}
