using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace PhoneCallsAnomalyDetection.DataSourses
{
    // define el numero de la columna en el archivo de datos

    public class PhoneCallsData
    {
        // Usado para cargar los datos de la columna 0 de timestamp
        [LoadColumn(0)]
        public string? timestamp;

        [LoadColumn(1)]
        public double value;

    }
}
