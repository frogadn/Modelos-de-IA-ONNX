using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace PhoneCallsAnomalyDetection.DataSourses
{
    // Especifica la clase de datos de predicción y los resultados de la detección de anomalías
    public class PhoneCallsPrediction
    {
        // Vector para contener los resultados de la detección de anomalías, incluidos isAnomaly, anomalyScore,
        // magnitud, expectedValue, boundaryUnits, upperBoundary y lowerBoundary.
        [VectorType(7)]
        public double[] Prediction { get; set; }
    }
}
