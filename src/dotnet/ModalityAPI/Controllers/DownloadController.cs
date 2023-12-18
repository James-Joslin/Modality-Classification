using Microsoft.AspNetCore.Mvc;
using System.IO;


[ApiController]
[Route("[controller]")]
public class ModelController : ControllerBase
{
    private readonly ModelService _modelService;

    public ModelController(ModelService modelService)
    {
        _modelService = modelService;
    }

    [HttpGet("download/{modelName}")]
    public IActionResult DownloadModel(string modelName)
    {
        var modelPath = _modelService.GetModelPath(modelName);
        if (string.IsNullOrEmpty(modelPath) || !System.IO.File.Exists(modelPath))
        {
            return NotFound("Model not found.");
        }

        var memory = new MemoryStream();
        using (var stream = new FileStream(modelPath, FileMode.Open))
        {
            stream.CopyTo(memory);
        }
        memory.Position = 0;
        return File(memory, "application/octet-stream", Path.GetFileName(modelPath));
    }

}
