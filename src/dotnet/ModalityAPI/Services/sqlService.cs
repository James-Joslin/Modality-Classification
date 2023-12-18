using System;
using Npgsql;
using System.IO;
using System.Threading.Tasks;
public class ScoreMetrics
{
    public int MaxPgsi { get; set; }
    public int MaxCore10 { get; set; }
    public int MaxReferralIndex { get; set; }  // Add this property
}


public partial class SqlService
{
    private readonly string _connectionString;
    private readonly Dictionary<string, string> _sqlPaths;

    public SqlService(IConfiguration configuration)
    {
        _sqlPaths = configuration.GetSection("SqlPaths").Get<Dictionary<string, string>>() ?? throw new InvalidOperationException("No SQL directories found") ;
        if (!_sqlPaths.Any())
        {
            throw new InvalidOperationException("SQL query files not loaded.");
        }
        _connectionString = Environment.GetEnvironmentVariable("SQLSERVER_CONNECTION_STRING") ?? throw new InvalidOperationException("Invalid connection string");
    }

    public async Task<T> ExecuteSqlQueryAsync<T>(string sqlQuery, Func<NpgsqlDataReader, Task<T>> processResult)
    {
        try
        {
            using var connection = new NpgsqlConnection(_connectionString);
            await connection.OpenAsync();
            using var command = new NpgsqlCommand(sqlQuery, connection);
            using var reader = await command.ExecuteReaderAsync();

            return await processResult(reader);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Error executing SQL query: {ex.Message}");
        }
    }

    public string GetSqlQueryPath(string key)
    {
        if (_sqlPaths.TryGetValue(key, out var path))
        {
            return path;
        }
        else
        {
            throw new ArgumentException($"SQL path for key '{key}' not found.");
        }
    }

}