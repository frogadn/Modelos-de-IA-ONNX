#region Documentación
/*
 * Este programa su finalidad es mostrar cómo realizar clasificación multiclase en un conjunto de datos de incidencias de soporte técnico.
 * Se utiliza un modelo de clasificación multiclase para clasificar las incidencias en diferentes áreas de soporte técnico.
 * 
 * 1. El programa carga los datos de un archivo de texto y los divide en conjuntos de entrenamiento y prueba.
 * 2. Construye y entrena un modelo de clasificación multiclase.
 * 3. Evalúa el modelo con el conjunto de datos de prueba.
 * 4. Realiza predicciones con el modelo.
 *
 * Referencias y documentación: https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/github-issue-classification
 *
 */
#endregion

using Microsoft.ML;

namespace IncidenciasSoporteTecnico_ClasificacionMulticlase
{

    class Program
    {
        // Obtiene la ruta del directorio de la aplicación
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);

        // Define la ruta del archivo de datos de entrenamiento
        private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Datos", "issues_train.tsv");

        // Define la ruta del archivo de datos de prueba
        private static string _testDataPath => Path.Combine(_appPath, "..", "..", "..", "Datos", "issues_test.tsv");

        // Define la ruta del archivo del modelo
        private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Model", "model.zip");

        private static MLContext _mlContext;
        private static PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;

        static void Main(string[] args)
        {
            // Inicializa el contexto de ML
            _mlContext = new MLContext(seed: 0);

            Console.WriteLine($"=============== Cargando conjunto de datos  ===============");

            // Carga los datos de entrenamiento desde un archivo de texto
            _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true);

            Console.WriteLine($"=============== Conjunto de datos cargado finalizado  ===============");

            // Procesa los datos
            var pipeline = ProcessData();

            // Construye y entrena el modelo
            var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);

            // Evalúa el modelo
            Evaluate(_trainingDataView.Schema);

            // Realiza una predicción de prueba
            PredictIssue();
        }

        public static IEstimator<ITransformer> ProcessData()
        {
            Console.WriteLine($"=============== Procesamiento de datos ===============");

            // Define el pipeline de procesamiento de datos
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                            .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                            .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                            .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                            .AppendCacheCheckpoint(_mlContext);

            Console.WriteLine($"=============== Datos de procesamiento finalizados ===============");

            return pipeline;
        }

        public static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            // Añade el algoritmo de clasificación al pipeline -------------------
            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                    .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Console.WriteLine($"=============== Entrenando el modelo  ===============");

            // Entrena el modelo
            _trainedModel = trainingPipeline.Fit(trainingDataView);

            Console.WriteLine($"=============== Finalizado el entrenamiento del modelo Hora de finalización: {DateTime.Now.ToString()} ===============");

            Console.WriteLine($"=============== Predicción única modelo recién entrenada ===============");

            // Crea un motor de predicción
            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_trainedModel);

            // Realiza una predicción de prueba
            GitHubIssue issue = new GitHubIssue()
            {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };

            var prediction = _predEngine.Predict(issue);

            Console.WriteLine($"=============== Predicción única del modelo recién entrenado - resultado: {prediction.Area} ===============");

            return trainingPipeline;
        }

        public static void Evaluate(DataViewSchema trainingDataViewSchema)
        {
            Console.WriteLine($"=============== Evaluación para obtener métricas de precisión del modelo - hora de inicio: {DateTime.Now.ToString()} ===============");

            // Carga los datos de prueba
            var testDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_testDataPath, hasHeader: true);

            // Evalúa el modelo con los datos de prueba
            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));

            Console.WriteLine($"=============== Evaluación para obtener métricas de precisión del modelo - Hora de finalización: {DateTime.Now.ToString()} ===============");
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");

            // Guarda el modelo entrenado
            SaveModelAsFile(_mlContext, trainingDataViewSchema, _trainedModel);
        }

        public static void PredictIssue()
        {
            // Carga el modelo guardado
            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

            // Crea un nuevo motor de predicción
            GitHubIssue singleIssue = new GitHubIssue() { Title = "Entity Framework crashes", Description = "When connecting to the database, EF is crashing" };
            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);

            // Realiza una predicción
            var prediction = _predEngine.Predict(singleIssue);

            Console.WriteLine($"*************************************************************************************************************");

            Console.WriteLine($"=============== Predicción única - Resultado: {prediction.Area} =========================");
        }

        private static void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            // Guarda el modelo en un archivo
            mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);

            Console.WriteLine("The model is saved to {0}", _modelPath);
        }
    }
}
