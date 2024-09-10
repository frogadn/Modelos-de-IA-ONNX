
#region Documentación
/*
 * Este programa su finalidad es la recomendación de películas mediante factorización matricial.
 * Se utiliza un modelo de factorización matricial para recomendar películas a los usuarios.
 * 
 * 1. El programa carga los datos de un archivo de texto y los divide en conjuntos de entrenamiento y prueba.
 * 2. Construye y entrena un modelo de factorización matricial.
 * 3. Evalúa el modelo con el conjunto de datos de prueba.
 * 4. Realiza una predicción individual con el modelo entrenado.
 *
 * Referencias y documentación: https://learn.microsoft.com/en-us/dotnet/machine-learning/tutorials/movie-recommendation
 *
 */
#endregion

using System.Data;
using System.Runtime.CompilerServices;
using Microsoft.ML;
using Microsoft.ML.Trainers;
//using MovieRecommendation;


namespace RecomendadorDePeliculas_FactorizacionMatricial
{

    namespace RecomendadorDePeliculas_FactorizacionMatricial
    {
        public class Program
        {
            static void Main(string[] args)
            {
                // Creamos un contexto de ML.NET
                MLContext mlContext = new MLContext();

                // Cargamos los datos de entrenamiento y prueba
                (IDataView trainingDataView, IDataView testDataView) = LoadData(mlContext);

                // Construimos y entrenamos el modelo
                ITransformer model = BuildAndTrainModel(mlContext, trainingDataView);

                // Evaluamos el modelo con los datos de prueba
                EvaluateModel(mlContext, testDataView, model);

                // Usamos el modelo para hacer una predicción individual
                UseModelForSinglePrediction(mlContext, model);

                // Guardamos el modelo entrenado en un archivo
                SaveModel(mlContext, trainingDataView.Schema, model);
            }

            // Método para cargar los datos de entrenamiento y prueba
            public static (IDataView training, IDataView test) LoadData(MLContext mlContext)
            {
                // Rutas de los archivos de datos
                var trainingDataPath = Path.Combine(Environment.CurrentDirectory, "Datos", "recommendation-ratings-train.csv");
                var testDataPath = Path.Combine(Environment.CurrentDirectory, "Datos", "recommendation-ratings-test.csv");

                // Cargamos los datos en IDataView
                IDataView trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating>(trainingDataPath, hasHeader: true, separatorChar: ',');
                IDataView testDataView = mlContext.Data.LoadFromTextFile<MovieRating>(testDataPath, hasHeader: true, separatorChar: ',');

                return (trainingDataView, testDataView);
            }

            // Método para construir y entrenar el modelo
            public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingDataView)
            {
                // Definimos el proceso de transformación de datos
                IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
                    .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: "movieId"));

                // Configuramos las opciones del entrenador de factorización matricial
                var options = new MatrixFactorizationTrainer.Options
                {
                    MatrixColumnIndexColumnName = "userIdEncoded",
                    MatrixRowIndexColumnName = "movieIdEncoded",
                    LabelColumnName = "Label",
                    NumberOfIterations = 20,
                    ApproximationRank = 100
                };

                // Añadimos el entrenador al proceso de transformación
                var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

                Console.WriteLine("=============== Entrenando el modelo ===============");
                // Entrenamos el modelo
                ITransformer model = trainerEstimator.Fit(trainingDataView);

                return model;
            }

            // Método para evaluar el modelo
            public static void EvaluateModel(MLContext mlContext, IDataView testDataView, ITransformer model)
            {
                Console.WriteLine("=============== Evaluación del modelo ===============");
                // Hacemos predicciones con los datos de prueba
                var prediction = model.Transform(testDataView);

                // Evaluamos las predicciones
                var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");

                // Mostramos las métricas de evaluación
                Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString()); // Error cuadrático medio Cuanto menor sea su valor, mejor será el modelo.
                Console.WriteLine("RSquared: " + metrics.RSquared.ToString()); // Le interesa que la puntuación de R Squared esté lo más cerca posible de 1.
            }

            // Método para usar el modelo en una predicción individual
            public static void UseModelForSinglePrediction(MLContext mlContext, ITransformer model)
            {
                Console.WriteLine("=============== Hacer una predicción ===============");
                // Creamos un motor de predicción
                var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);

                // Definimos una entrada de prueba ---------
                var testInput = new MovieRating { userId = 3, movieId = 10 };

                // Hacemos la predicción
                var movieRatingPrediction = predictionEngine.Predict(testInput);

                // Mostramos el resultado de la predicción
                if (Math.Round(movieRatingPrediction.Score, 1) > 3.5)
                {
                    Console.WriteLine("Movie " + testInput.movieId + " is recommended for user " + testInput.userId);
                }
                else
                {
                    Console.WriteLine("Movie " + testInput.movieId + " is not recommended for user " + testInput.userId);
                }
            }

            // Método para guardar el modelo entrenado en un archivo
            public static void SaveModel(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
            {
                // Ruta donde se guardará el modelo
                var modelPath = Path.Combine(Environment.CurrentDirectory, "C:\\Frog\\Modelos de IA ONNX\\RecomendadorDePeliculas_FactorizacionMatricial\\RecomendadorDePeliculas_FactorizacionMatricial\\Model\\MovieRecommenderModel.zip");

                Console.WriteLine("=============== Guardar el modelo en un archivo ===============");
                // Guardamos el modelo
                mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
            }
        }
    }
}