using System;
using Npgsql;
using System.IO;
using System.Threading.Tasks;

public partial class SqlService
{
    public static async Task<ScoreMetrics> ProcessMinMaxMetrics(NpgsqlDataReader reader)
    {
        var scoreMetrics = new ScoreMetrics();
        while (await reader.ReadAsync())
        {
            string scoreType = reader.GetString(0);
            int maxScore = reader.GetInt32(2);

            switch (scoreType)
            {
                case "PGSI":
                    scoreMetrics.MaxPgsi = maxScore;
                    break;
                case "CORE10":
                    scoreMetrics.MaxCore10 = maxScore;
                    break;
                case "Ref_Index":
                    scoreMetrics.MaxReferralIndex = maxScore;
                    break;
            }
        }
        return scoreMetrics;
    }

    public static async Task<string> ProcessModalityString(NpgsqlDataReader reader)
    {
        if (await reader.ReadAsync())
        {
            return reader.GetString(0); // Assuming the result is a string at column 0
        }
        return string.Empty; // Or handle appropriately
    }

    // Other processing methods as needed
}
