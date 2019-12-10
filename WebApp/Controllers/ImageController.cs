using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.ML;
using WebApp.Infrastructure;
using WebApp.ML.DataModels;

namespace WebApp.Controllers
{
    public class ImageController : Controller
    {
        public IConfiguration Configuration { get; }
        private readonly PredictionEnginePool<ImageInputData, ImageLabelPredictions> _predictionEnginePool;
        private readonly ILogger<ImageController> _logger;
        private readonly string _labelsFilePath;

        public ImageController(PredictionEnginePool<ImageInputData, ImageLabelPredictions> predictionEnginePool, IConfiguration configuration, ILogger<ImageController> logger)
        {
            // Get the ML Model Engine injected, for scoring
            _predictionEnginePool = predictionEnginePool;

            Configuration = configuration;
            _labelsFilePath = GetAbsolutePath(Configuration["MLModel:LabelsFilePath"]);

            //Get other injected dependencies
            _logger = logger;
        }
        [HttpGet]
        public IActionResult Index()
        {
            return View();
        }
        [HttpPost]
        public async Task<IActionResult> Index(IFormFile imageFile)
        {
            if (imageFile.Length == 0)
                return BadRequest();

            MemoryStream imageMemoryStream = new MemoryStream();
            await imageFile.CopyToAsync(imageMemoryStream);

            //Check that the image is valid
            byte[] imageData = imageMemoryStream.ToArray();
            if (!imageData.IsValidImage())
                return StatusCode(StatusCodes.Status415UnsupportedMediaType);

            //Convert to Image
            Image image = Image.FromStream(imageMemoryStream);

            //Convert to Bitmap
            Bitmap bitmapImage = (Bitmap)image;

            _logger.LogInformation($"Start processing image...");

            //Measure execution time
            var watch = System.Diagnostics.Stopwatch.StartNew();

            //Set the specific image data into the ImageInputData type used in the DataView
            ImageInputData imageInputData = new ImageInputData { Image = bitmapImage };

            //Predict code for provided image
            ImageLabelPredictions imageLabelPredictions = _predictionEnginePool.Predict(imageInputData);

            //Stop measuring time
            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;
            _logger.LogInformation($"Image processed in {elapsedMs} miliseconds");

            //Predict the image's label (The one with highest probability)
            ImagePredictedLabelWithProbability imageBestLabelPrediction
                                = FindBestLabelWithProbability(imageLabelPredictions, imageInputData);
            ViewBag.ImagePredicted = imageBestLabelPrediction;
            return View();
        }

        private ImagePredictedLabelWithProbability FindBestLabelWithProbability(ImageLabelPredictions imageLabelPredictions, ImageInputData imageInputData)
        {
            //Read TF model's labels (labels.txt) to classify the image across those labels
            var labels = ReadLabels(_labelsFilePath);

            float[] probabilities = imageLabelPredictions.PredictedLabels;

            //Set a single label as predicted or even none if probabilities were lower than 70%
            var imageBestLabelPrediction = new ImagePredictedLabelWithProbability()
            {
                ImageId = imageInputData.GetHashCode().ToString(), //This ID is not really needed, it could come from the application itself, etc.
            };

            (imageBestLabelPrediction.PredictedLabel, imageBestLabelPrediction.Probability) = GetBestLabel(labels, probabilities);

            return imageBestLabelPrediction;
        }

        private (string, float) GetBestLabel(string[] labels, float[] probs)
        {
            var max = probs.Max();
            var index = probs.AsSpan().IndexOf(max);
            var maxIndex = max.ToString();
            if (maxIndex.StartsWith("0.")) maxIndex = maxIndex.Remove(0, 2);
            if (maxIndex.Length > 3) maxIndex = maxIndex.Substring(0,3);

            var tmpIndex = Convert.ToInt32(maxIndex);

            if (max > 0.7)
                return (labels[tmpIndex], max);
            else
                return ("None", max);
        }

        private string[] ReadLabels(string labelsLocation)
        {
            return System.IO.File.ReadAllLines(labelsLocation);
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);
            return fullPath;
        }
    }
}