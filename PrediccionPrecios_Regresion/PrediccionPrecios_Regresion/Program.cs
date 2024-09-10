#region Documentación
/*
 * Este programa su finalidad es mostrar cómo realizar predicciones de precios de taxis mediante regresión.
 * Se utiliza un modelo de regresión para predecir la tarifa de un viaje en taxi.
 * 
 * 1. El programa carga los datos de un archivo de texto y entrena un modelo de regresión.
 * 2. Evalúa el modelo con un conjunto de datos de prueba.
 * 3. Realiza una predicción única con el modelo entrenado.
 * 4. Imprime los resultados de la predicción.
 *
 * Referencias y documentación: https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/predict-prices
 *
 */
#endregion

using Microsoft.ML;

namespace PrediccionPrecios_Regresion
{
    class Program
    {
        /// <summary>
        /// Ruta del archivo de datos de entrenamiento.
        /// </summary>
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Datos", "taxi-fare-train.csv");

        /// <summary>
        /// Ruta del archivo de datos de prueba.
        /// </summary>
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Datos", "taxi-fare-test.csv");

        /// <summary>
        /// Ruta del archivo del modelo entrenado.
        /// </summary>
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Datos", "Model.zip");

        /// <summary>
        /// Punto de entrada principal del programa.
        /// </summary>
        /// <param name="args">Argumentos de la línea de comandos.</param>
        static void Main(string[] args)
        {
            Console.WriteLine(Environment.CurrentDirectory);

            // Crear el contexto de ML
            MLContext mlContext = new MLContext(seed: 0);

            // Entrenar el modelo
            var model = Train(mlContext, _trainDataPath);

            // Evaluar el modelo
            Evaluate(mlContext, model);

            // Probar una predicción única
            TestSinglePrediction(mlContext, model);
        }

        /// <summary>
        /// Entrena el modelo de regresión.
        /// </summary>
        /// <param name="mlContext">El contexto de ML.</param>
        /// <param name="dataPath">La ruta del archivo de datos de entrenamiento.</param>
        /// <returns>El modelo entrenado.</returns>
        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            // Cargar los datos de entrenamiento
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: ',');

            // Definir el pipeline de entrenamiento
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
                .Append(mlContext.Regression.Trainers.FastTree());

            Console.WriteLine("=============== Crear y Entrenar el Modelo ===============");

            // Entrenar el modelo
            var model = pipeline.Fit(dataView);

            Console.WriteLine("=============== Fin del entrenamiento ===============");
            Console.WriteLine();

            return model;
        }

        /// <summary>
        /// Evalúa el modelo entrenado.
        /// </summary>
        /// <param name="mlContext">El contexto de ML.</param>
        /// <param name="model">El modelo entrenado.</param>
        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            // Cargar los datos de prueba
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');

            // Hacer predicciones
            var predictions = model.Transform(dataView);

            // Evaluar las predicciones
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Evaluación de métricas de calidad del modelo         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       Puntuación RSquared:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Error Cuadrático Medio:      {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*************************************************");
        }

        /// <summary>
        /// Prueba una predicción única con el modelo entrenado.
        /// </summary>
        /// <param name="mlContext">El contexto de ML.</param>
        /// <param name="model">El modelo entrenado.</param>
        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            // Crear el motor de predicción
            var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);

            // Crear una muestra de datos de taxi
            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // Para predecir. Real/Observado = 15.5
            };

            // Hacer la predicción
            var prediction = predictionFunction.Predict(taxiTripSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Tarifa predicha: {prediction.FareAmount:0.####}, tarifa real: 15.5");
            Console.WriteLine($"**********************************************************************");
        }
    }
}

