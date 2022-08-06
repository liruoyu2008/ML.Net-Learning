using Microsoft.ML;
using Microsoft.ML.TimeSeries;
using PhoneCallsAnomalyDetection;

string _dataPath = Path.Combine(Environment.CurrentDirectory, "../../../Data", "phone-calls.txt");
MLContext mlContext = new MLContext();
IDataView dataView = mlContext.Data.LoadFromTextFile<PhoneCallsData>(_dataPath, hasHeader: true, separatorChar: ',');
int period = DetectPeriod(mlContext, dataView);
DetectAnomaly(mlContext, dataView, period);


int DetectPeriod(MLContext mlContext, IDataView phoneCalls)
{
    int period = mlContext.AnomalyDetection.DetectSeasonality(phoneCalls, nameof(PhoneCallsData.value));
    Console.WriteLine("Period of the series is: {0}.", period);
    return period;
}

void DetectAnomaly(MLContext mlContext, IDataView phoneCalls, int period)
{
    var options = new SrCnnEntireAnomalyDetectorOptions()
    {
        Threshold = 0.3,
        Sensitivity = 64.0,
        DetectMode = SrCnnDetectMode.AnomalyAndMargin,
        Period = period,
    };
    var outputDataView = mlContext.AnomalyDetection
        .DetectEntireAnomalyBySrCnn(phoneCalls, nameof(PhoneCallsPrediction.Prediction), nameof(PhoneCallsData.value), options);
    var predictions = mlContext.Data.CreateEnumerable<PhoneCallsPrediction>(
    outputDataView, reuseRowObject: false);
    Console.WriteLine("Index,Data,Anomaly,AnomalyScore,Mag,ExpectedValue,BoundaryUnit,UpperBoundary,LowerBoundary");

    var index = 0;

    foreach (var p in predictions)
    {
        if (p.Prediction[0] == 1)
        {
            Console.WriteLine("{0},{1},{2},{3},{4},  <-- alert is on! detected anomaly", index,
                p.Prediction[0], p.Prediction[3], p.Prediction[5], p.Prediction[6]);
        }
        else
        {
            Console.WriteLine("{0},{1},{2},{3},{4}", index,
                p.Prediction[0], p.Prediction[3], p.Prediction[5], p.Prediction[6]);
        }
        ++index;

    }

    Console.WriteLine("");
}
