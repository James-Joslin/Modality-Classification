using Microsoft.ML;
using Microsoft.ML.Data;

var connectionString = Environment.GetEnvironmentVariable("SQLSERVER_CONNECTION_STRING");
var builder = WebApplication.CreateBuilder(args);
// add services
builder.Services.AddSingleton<ModelService>();
builder.Services.AddSingleton<SqlService>();


// Add services to the container. # CORS redirection to use specified domains and HTTPS when domain is set
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(
        builder =>
        {
            builder.AllowAnyOrigin() // Adjust the ports accordingly when a front-end domain is actually specified/built
                   .AllowAnyHeader()
                   .AllowAnyMethod();
        });
});


builder.Services.AddControllers();
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
    // app.UseCors(); # Enabled when domain is set up
}

app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

app.Run();
