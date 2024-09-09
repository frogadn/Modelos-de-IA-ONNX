using Microsoft.ML;
using Microsoft.ML.TimeSeries;
using PhoneCallsAnomalyDetection;
using PhoneCallsAnomalyDetection.DataSourses;

namespace PhoneCallsAnomalyDetection
{
    public class Program
    {

        // Ruta de acceso al archivo de datos
        static string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "C:\\Frog\\ModelosIA\\Modelos IA\\PruebaDeteccionAnomalias\\PruebaDeteccionAnomalias\\Datos\\phone-calls.csv");


        static void Main(string[] args)
        {
            // crear el MLContext para compartir entre los componentes del flujo de trabajo de creación del modelo
            MLContext mlcontext = new MLContext();

            // PASO 1: Configuración de carga de datos comunes
            // Este dataset es usado para detectar anomalías en las llamadas telefónicas

            IDataView dataView = mlcontext.Data.LoadFromTextFile<PhoneCallsData>(path: _dataPath, hasHeader: true, separatorChar: ',');

            // Detectar el período de la serie 
            int period = DetectPeriod(mlcontext, dataView);

            // Detectar anomalías en la serie con la información del periodo 
            DetectAnomaly(mlcontext, dataView, period);

        }

        public static int DetectPeriod(MLContext mlContext, IDataView phoneCalls)
        {

            Console.WriteLine("Detectar el período de la serie");

            // Detectar el periodo de la serie 

            int period = mlContext.AnomalyDetection.DetectSeasonality(phoneCalls, nameof(PhoneCallsData.value));

            // Imprime el periodo de la serie 
            Console.WriteLine("\nPeriodo de la serie {0}.", period);

            return period;
        }

        // Detectar anomalías en la serie con la información del periodo
        public static void DetectAnomaly(MLContext mlContext, IDataView phoneCalls, int period)
        {
           
            var options = new SrCnnEntireAnomalyDetectorOptions()
            {
                Threshold = 0.3,
                Sensitivity = 64.0,
                DetectMode = SrCnnDetectMode.AnomalyAndMargin,
                Period = period,
            };

            // Detectar anomalías en la serie con la información del periodo

            var outputDataView = mlContext.AnomalyDetection.DetectEntireAnomalyBySrCnn(phoneCalls, nameof(PhoneCallsPrediction.Prediction), nameof(PhoneCallsData.value), options);

            // Convertir la salida en una lista
            var predictions = mlContext.Data.CreateEnumerable<PhoneCallsPrediction>(outputDataView, reuseRowObject: false);

            // Encabezado de la salida
            Console.WriteLine("\nIndex\tData\tBoundaryUnit\tUpperBoundary\tLowerBoundary");

            // Incdes = indice de cada punto, Anomaly= si es una anomalía, AnomalyScore = puntuación de la anomalía, Mag = magnitud,
            // ExpectedValue = valor esperado, BoundaryUnit = unidad de límite, UpperBoundary = límite superior, LowerBoundary = límite inferior
            var index = 0;

            // iterar sobre las predicciones para imprimir los resultados
            foreach (var p in predictions)
            {
                if (p.Prediction is not null)
                {
                    string output;
                    if (p.Prediction[0] == 1)
                        output = "{0},{1},{2},{3},{4},  <-- Alerta prendida! Detección de anomalía";
                    else
                        output = "{0},{1},{2},{3},{4}";

                    Console.WriteLine(output, index, p.Prediction[0], p.Prediction[3], p.Prediction[5], p.Prediction[6]);
                }
                ++index;
            }

            Console.WriteLine("");
        }

    }
}
