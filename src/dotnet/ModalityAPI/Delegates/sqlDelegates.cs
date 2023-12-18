using System;
using Npgsql;
using System.IO;
using System.Threading.Tasks;

public delegate Task<ScoreMetrics> ProcessMinMaxMetrics(NpgsqlDataReader reader);
public delegate Task<string> ProcessModalityString(NpgsqlDataReader reader);
// Other delegates as needed
