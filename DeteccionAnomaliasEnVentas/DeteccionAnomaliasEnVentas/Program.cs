using Microsoft.ML;

namespace DeteccionAnomaliasEnVentas
{
    public class Program
    
    {
        /// <summary>
        /// Ruta del archivo de datos CSV.
        /// </summary>
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "C:\\Frog\\Modelos de IA ONNX\\DeteccionAnomaliasEnVentas\\DeteccionAnomaliasEnVentas\\Datos", "product-sales.csv");

        /// <summary>
        /// Número de registros en el archivo de datos.
        /// </summary>
        const int _docsize = 36;

        /// <summary>
        /// Método principal del programa.
        /// </summary>
        /// <param name="args">Argumentos de línea de comandos.</param>
        static void Main(string[] args)
        {
            // Crear MLContext para ser compartido a lo largo del flujo de trabajo de creación del modelo.
            MLContext mlContext = new MLContext();

            // Configuración común de carga de datos.
            IDataView dataView = mlContext.Data.LoadFromTextFile<ProductSalesData>(path: _dataPath, hasHeader: true, separatorChar: ',');

            // Detectar cambios temporales en el patrón.
            DetectSpike(mlContext, _docsize, dataView);

            // Detectar cambios persistentes en el patrón.
            DetectChangepoint(mlContext, _docsize, dataView);
        }

        /// <summary>
        /// Detecta cambios temporales en el patrón de ventas.
        /// </summary>
        /// <param name="mlContext">Contexto de ML.NET.</param>
        /// <param name="docSize">Tamaño del documento.</param>
        /// <param name="productSales">Datos de ventas de productos.</param>
        static void DetectSpike(MLContext mlContext, int docSize, IDataView productSales)
        {
            Console.WriteLine("Detectar cambios temporales en el patrón");

            // Configurar el algoritmo de entrenamiento.
            var iidSpikeEstimator = mlContext.Transforms.DetectIidSpike(outputColumnName: nameof(ProductSalesPrediction.Prediction), inputColumnName: nameof(ProductSalesData.numSales), confidence: 95, pvalueHistoryLength: docSize / 4);

            // Crear la transformación de detección de picos.
            Console.WriteLine("=============== Entrenando el modelo ===============");
            ITransformer iidSpikeTransform = iidSpikeEstimator.Fit(CreateEmptyDataView(mlContext));
            Console.WriteLine("=============== Fin del proceso de entrenamiento ===============");

            // Aplicar la transformación de datos para crear predicciones.
            IDataView transformedData = iidSpikeTransform.Transform(productSales);

            // Crear una enumeración de predicciones.
            var predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);

            // Mostrar los resultados.
            Console.WriteLine("Alerta\tPuntuación\tValor-P");
            foreach (var p in predictions)
            {
                var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}";

                if (p.Prediction[0] == 1)
                {
                    results += " <-- Pico detectado";
                }

                Console.WriteLine(results);
            }
            Console.WriteLine("");
        }

        /// <summary>
        /// Detecta cambios persistentes en el patrón de ventas.
        /// </summary>
        /// <param name="mlContext">Contexto de ML.NET.</param>
        /// <param name="docSize">Tamaño del documento.</param>
        /// <param name="productSales">Datos de ventas de productos.</param>
        static void DetectChangepoint(MLContext mlContext, int docSize, IDataView productSales)
        {
            Console.WriteLine("Detectar cambios persistentes en el patrón");

            // Configurar el algoritmo de entrenamiento.
            var iidChangePointEstimator = mlContext.Transforms.DetectIidChangePoint(outputColumnName: nameof(ProductSalesPrediction.Prediction), inputColumnName: nameof(ProductSalesData.numSales), confidence: 95, changeHistoryLength: docSize / 4);

            // Crear la transformación de detección de puntos de cambio.
            Console.WriteLine("=============== Entrenando el modelo usando el algoritmo de detección de puntos de cambio ===============");
            var iidChangePointTransform = iidChangePointEstimator.Fit(CreateEmptyDataView(mlContext));
            Console.WriteLine("=============== Fin del proceso de entrenamiento ===============");

            // Aplicar la transformación de datos para crear predicciones.
            IDataView transformedData = iidChangePointTransform.Transform(productSales);

            // Crear una enumeración de predicciones.
            var predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);

            // Mostrar los resultados.
            Console.WriteLine("Alerta\tPuntuación\tValor-P\tValor de Martingala");
            foreach (var p in predictions)
            {
                var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}\t{p.Prediction[3]:F2}";

                if (p.Prediction[0] == 1)
                {
                    results += " <-- alerta activada, punto de cambio detectado";
                }
                Console.WriteLine(results);
            }
            Console.WriteLine("");
        }

        /// <summary>
        /// Crea un DataView vacío. Solo necesitamos el esquema para llamar a Fit() para las transformaciones de series temporales.
        /// </summary>
        /// <param name="mlContext">Contexto de ML.NET.</param>
        /// <returns>Un IDataView vacío.</returns>
        static IDataView CreateEmptyDataView(MLContext mlContext)
        {
            IEnumerable<ProductSalesData> enumerableData = new List<ProductSalesData>();
            return mlContext.Data.LoadFromEnumerable(enumerableData);
        }
    }
}
