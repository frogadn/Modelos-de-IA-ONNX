﻿using System;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
//using Microsoft.ML.OnnxRuntime.DirectML;

using System.Drawing;
using System.Drawing.Drawing2D;
//using ObjectDetection.YoloParser;
//using ObjectDetection.DataStructures;
//using ObjectDetection;
using Microsoft.ML;
using PruebaModelosIA;
using YoloParser;

var assetsRelativePath = @"../../../assets";
string assetsPath = GetAbsolutePath(assetsRelativePath);
var modelFilePath = Path.Combine(assetsPath, "Model", "TinyYolo2_model.onnx");
var imagesFolder = Path.Combine(assetsPath, "images");
var outputFolder = Path.Combine(assetsPath, "images", "output");

// Initialize MLContext
MLContext mlContext = new MLContext();

try
{
    // Load Data
    IEnumerable<ImageNetData> images = ImageNetData.ReadFromFile(imagesFolder);
    IDataView imageDataView = mlContext.Data.LoadFromEnumerable(images);

    // Create instance of model scorer
    var modelScorer = new OnnxModelScorer(imagesFolder, modelFilePath, mlContext);

    // Use model to score data
    IEnumerable<float[]> probabilities = modelScorer.Score(imageDataView);

    // Post-process model output
    YoloOutputParser parser = new YoloOutputParser();

    var boundingBoxes =
        probabilities
        .Select(probability => parser.ParseOutputs(probability))
        .Select(boxes => parser.FilterBoundingBoxes(boxes, 5, .5F));

    // Draw bounding boxes for detected objects in each of the images
    for (var i = 0; i < images.Count(); i++)
    {
        string imageFileName = images.ElementAt(i).Label;
        IList<YoloBoundingBox> detectedObjects = boundingBoxes.ElementAt(i);

        DrawBoundingBox(imagesFolder, outputFolder, imageFileName, detectedObjects);

        LogDetectedObjects(imageFileName, detectedObjects);
    }
}
catch (Exception ex)
{
    Console.WriteLine(ex.ToString());
}

Console.WriteLine("========= Termino el proceso ========");
string GetAbsolutePath(string relativePath)
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;

    string fullPath = Path.Combine(assemblyFolderPath, relativePath);

    return fullPath;
}


void DrawBoundingBox(string inputImageLocation, string outputImageLocation, string imageName, IList<YoloBoundingBox> filteredBoundingBoxes)
{
    Image image = Image.FromFile(Path.Combine(inputImageLocation, imageName));

    var originalImageHeight = image.Height;
    var originalImageWidth = image.Width;

    foreach (var box in filteredBoundingBoxes)
    {
        var x = (uint)Math.Max(box.Dimensions.X, 0);
        var y = (uint)Math.Max(box.Dimensions.Y, 0);
        var width = (uint)Math.Min(originalImageWidth - x, box.Dimensions.Width);
        var height = (uint)Math.Min(originalImageHeight - y, box.Dimensions.Height);

        x = (uint)originalImageWidth * x / OnnxModelScorer.ImageNetSettings.imageWidth;
        y = (uint)originalImageHeight * y / OnnxModelScorer.ImageNetSettings.imageHeight;
        width = (uint)originalImageWidth * width / OnnxModelScorer.ImageNetSettings.imageWidth;
        height = (uint)originalImageHeight * height / OnnxModelScorer.ImageNetSettings.imageHeight;

        string text = $"{box.Label} ({(box.Confidence * 100).ToString("0")}%)";

        using (Graphics thumbnailGraphic = Graphics.FromImage(image))
        {
            thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
            thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
            thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;

            // Define Text Options
            Font drawFont = new Font("Arial", 12, FontStyle.Bold);
            SizeF size = thumbnailGraphic.MeasureString(text, drawFont);
            SolidBrush fontBrush = new SolidBrush(Color.Black);
            Point atPoint = new Point((int)x, (int)y - (int)size.Height - 1);

            // Define BoundingBox options
            Pen pen = new Pen(box.BoxColor, 3.2f);
            SolidBrush colorBrush = new SolidBrush(box.BoxColor);

            thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);


            thumbnailGraphic.DrawString(text, drawFont, fontBrush, atPoint);

            // Draw bounding box on image
            thumbnailGraphic.DrawRectangle(pen, x, y, width, height);

        }


    }
    if (!Directory.Exists(outputImageLocation))
    {
        Directory.CreateDirectory(outputImageLocation);
    }

    image.Save(Path.Combine(outputImageLocation, imageName));
}

void LogDetectedObjects(string imageName, IList<YoloBoundingBox> boundingBoxes)
{
    Console.WriteLine($".....Los objetos identificados en la imagen {imageName} se muestra a continuación....");

    foreach (var box in boundingBoxes)
    {
        Console.WriteLine($"{box.Label} y su puntuación de confianza: {box.Confidence}");
    }

    // and its Confidence score:

    Console.WriteLine("");

}


//namespace PruebaModelosIA
//{
//    class Program
//    {


//        static void Main(string[] args)
//        {
//            // Ruta al modelo ONNX
//            string modelPath = "C:/Frog/ModelosIA/ModelosONNX/models-main/validated/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-7.onnx";

//            // Crear opciones de sesión y añadir el proveedor de ejecución DirectML
//            var sessionOptions = new SessionOptions();
//            sessionOptions.AppendExecutionProvider_DML();

//            // Crear la sesión de ONNX Runtime
//            using var session = new InferenceSession(modelPath, sessionOptions);

//            // Preparar los datos de entrada (esto variará según tu modelo)
//            var inputTensor = new DenseTensor<float>(new float[] { /* tus datos aquí */ }, new int[] { 1, 3, 224, 224 });
//            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inputTensor) };

//            // Ejecutar el modelo
//            using var results = session.Run(inputs);

//            // Procesar los resultados
//            foreach (var result in results)
//            {
//                Console.WriteLine($"Resultado: {result.Name}: {result.AsTensor<float>().GetArrayString()}");
//            }
//        }
//    }

//}
