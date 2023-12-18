using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;

using Microsoft.AspNetCore.Mvc;
using System.IO;
using System.Threading.Tasks;



[ApiController]
[Route("[controller]")]
public class PredictionController : ControllerBase
{
    private readonly SqlService _sqlService;
    private readonly ModelService _modelService;

    public PredictionController(SqlService sqlService, ModelService modelService)
    {
        _sqlService = sqlService;
        _modelService = modelService;
    }

    [HttpPost("predict")]
    public async Task<IActionResult> Predict([FromBody] PredictionRequest request)
    {
        // Retrieve the maximum scores and max referral index
        // string queryFilePath = Path.Combine(Directory.GetCurrentDirectory(), "SqlQueries", "minimax_metrics.sql");
        string sqlQuery = System.IO.File.ReadAllText(_sqlService.GetSqlQueryPath("minimax"));
        ScoreMetrics metrics = await _sqlService.ExecuteSqlQueryAsync(sqlQuery, SqlService.ProcessMinMaxMetrics);

        // Validate the request values
        if (request.Pgsi > metrics.MaxPgsi || request.Core10 > metrics.MaxCore10)
        {
            return BadRequest("Input PGSI and/or Core10 values exceed maximum allowed scores.");
        }

        // Normalize pgsi and core10
        float normalizedPgsi = request.Pgsi / metrics.MaxPgsi;
        float normalizedCore10 = request.Core10 / metrics.MaxCore10;

        // One-hot encode referralIndex (0-based)
        if (request.ReferralIndex < 0 || request.ReferralIndex >= 30)
        {
            return BadRequest("ReferralIndex is out of range.");
        }

        int[] referralVector = new int[metrics.MaxReferralIndex + 1]; 
        referralVector[request.ReferralIndex] = 1; // Directly use the 0-based index

        // Combine into a single vector
        var inputVector = new List<float> { normalizedPgsi, normalizedCore10 };
        inputVector.AddRange(referralVector.Select(x => (float)x));

        // Run the model with the input vector
        var outputVectors = _modelService.RunModel("brackets", inputVector);

        var extendedVector = new List<float>(inputVector);
        foreach (var vector in outputVectors)
        {
            extendedVector.AddRange(vector);
        }

        var modalityVector = _modelService.RunModel("modality", extendedVector);
        int modality_index = modalityVector[0].ToList().IndexOf(modalityVector[0].Max());

        // queryFilePath = Path.Combine(Directory.GetCurrentDirectory(), "SqlQueries", "get_modality_string_by_index.sql");
        sqlQuery = System.IO.File.ReadAllText(_sqlService.GetSqlQueryPath("modality_lookup")).Replace("@elementId", modality_index.ToString());
        string modalityString = await _sqlService.ExecuteSqlQueryAsync(sqlQuery, SqlService.ProcessModalityString);

        var response = new Dictionary<int, string>
        {
            { modality_index, modalityString }
        };

        return Ok(response);
    }
}
