#region Documentación
/*
 * Este programa su finalidad es detectar objetos en imágenes mediante transferencia de aprendizaje.
 * Se utiliza un modelo de detección de objetos para detectar objetos en imágenes.
 * 
 * 1. El programa carga los datos de un archivo de imágenes y los procesa con un modelo de detección de objetos.
 * 2. Realiza la detección de objetos en las imágenes.
 * 3. Dibuja cuadros delimitadores alrededor de los objetos detectados.
 * 4. Imprime los objetos detectados en cada imagen.
 *
 * Referencias y documentación: https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/image-classification
 *
 */
#endregion

using Microsoft.ML.Data;
using Microsoft.ML;

namespace TransferLearningTF
{
    class Program
    {
        // declaración de variables globales
        static readonly string _assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");  // carpeta de archivos
        static readonly string _imagesFolder = Path.Combine(_assetsPath, "images"); // carpeta de imágenes
        static readonly string _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv"); // archivo de entrenamiento
        static readonly string _testTagsTsv = Path.Combine(_imagesFolder, "test-tags.tsv"); // archivo de prueba
        static readonly string _predictSingleImage = Path.Combine(_imagesFolder, "teddy2.jpg"); // una imagen para predecir --------------------------
        static readonly string _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");
        

        static void Main(string[] args)
        {
            // Create MLContext to be shared across the model creation workflow objects
            MLContext mlContext = new MLContext();

           // llamada a la función GenerateModel
            ITransformer model = GenerateModel(mlContext);

            // llamada a la función ClassifySingleImage para predecir una imagen
            ClassifySingleImage(mlContext, model);

            Console.ReadKey();
        }

        // construye y genera el modelo
        public static ITransformer GenerateModel(MLContext mlContext)
        {
            // se agregan los estimadores para cargar, cambiar el tamaño y extraer píxeles de las imágenes
            IEstimator<ITransformer> pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _imagesFolder, inputColumnName: nameof(ImageData.ImagePath))
                    // La imagen transforma las imágenes al formato esperado del modelo.
                    .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
                    .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
                    // La transformación ScoreTensorFlowModel puntúa el modelo TensorFlow y permite la comunicación
                    // agrega eel estimador para cargar el modelo TensorFlow
                    .Append(mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel).
                        ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
                    // estimador para asignar las etiquetas de cadena en los datos de entrenamiento a los valores de clave entera:
                    .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
                    // Agregue el algoritmo de aprendizaje de ML.NET:
                    .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation"))
                    .AppendCacheCheckpoint(mlContext)
                    // Agrega el estimador para asignar el valor de clave de predicción a una cadena:
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"));

            // carga los datos de entrenamiento y entrena el modelo 
            IDataView trainingData = mlContext.Data.LoadFromTextFile<ImageData>(path: _trainTagsTsv, hasHeader: false);

            // muestra el modelo de clasificación de entrenamiento
            Console.WriteLine("\n=============== Modelo de clasificación de entrenamiento ===============");

            //  llama al método ya entrenado modelo
            ITransformer model = pipeline.Fit(trainingData);
        
            // genera predicciones a partir de los datos de prueba, para ser evaluados
            IDataView testData = mlContext.Data.LoadFromTextFile<ImageData>(path: _testTagsTsv, hasHeader: false);
            IDataView predictions = model.Transform(testData);

            // crea un IEnumerable para las predicciones para mostrar los resultados
            IEnumerable<ImagePrediction> imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true);
            DisplayResults(imagePredictionData);

            // muestra las métricas de clasificación
            Console.WriteLine("\n=============== Métricas de clasificación ===============");

            // evaluación del modelo 
            MulticlassClassificationMetrics metrics =
                mlContext.MulticlassClassification.Evaluate(predictions,
                  labelColumnName: "LabelKey",
                  predictedLabelColumnName: "PredictedLabel");
          
            // muestra las métricas de evaluación
            // proporción de predicciones correctas en comparación con todas las predicciones
            Console.WriteLine($"\nPerdida de registros: {metrics.LogLoss}"); // La pérdida de registro es una medida de la precisión de un clasificador, entre más bajo mejor
            // pérdida logarítmica por clase esté lo más cerca posible de 0
            Console.WriteLine($"\nPerdida logarítmica: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");

            return model;
        }

        // método para predecir una imagen 
        public static void ClassifySingleImage(MLContext mlContext, ITransformer model)
        {
            // Cargue el nombre de archivo de imagen completo en ImageData
            var imageData = new ImageData()
            {
                // nombre de la imagen  
                ImagePath = _predictSingleImage
            };

            // Realizar función de predicción (entrada = ImageData, salida = ImagePrediction)
            var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            var prediction = predictor.Predict(imageData);

            Console.WriteLine("\n=============== Realizar una clasificación de imágenes individuales ===============");

            // Mostrar los resultados de la predicción
            Console.WriteLine($"\nImagen: {Path.GetFileName(imageData.ImagePath)} clasificada como: {prediction.PredictedLabelValue} con una confianza de: {prediction.Score.Max()} ");
        }
        

        // método para mostrar los resultados
        private static void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
        {
            // visualiza los resultados de las predicciones 
            foreach (ImagePrediction prediction in imagePredictionData)
            {
                Console.WriteLine($"\nImagen: {Path.GetFileName(prediction.ImagePath)} clasificada como: {prediction.PredictedLabelValue} con una confianza de: {prediction.Score.Max()} ");
            }
        }

        // asigna los valores de las variables de entrada
        private struct InceptionSettings
        {
            public const int ImageHeight = 224; // altura de la imagen  
            public const int ImageWidth = 224; // ancho de la imagen
            public const float Mean = 117; // media
            public const float Scale = 1; // escala
            public const bool ChannelsLast = true; // canales
        }
        // clase para los datos de entrada y las predicciones 
        public class ImageData
        {
            [LoadColumn(0)]
            public string ImagePath; // ruta de la imagen

            [LoadColumn(1)]
            public string Label; // etiqueta de la imagen
        }
        
        // clase de predicción
        public class ImagePrediction : ImageData
        {
            public float[] Score; // porcentaje de confianza

            public string PredictedLabelValue; // etiqueta predicha
        }
    }
}
