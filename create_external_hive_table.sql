CREATE DATABASE IF NOT EXISTS game_analytics;
USE game_analytics;

CREATE EXTERNAL TABLE IF NOT EXISTS player_churn_raw (
    PlayerID INT,
    Age INT,
    Gender STRING,
    Location STRING,
    GameGenre STRING,
    PlayTimeHours DOUBLE,
    InGamePurchases INT,
    GameDifficulty STRING,
    SessionsPerWeek INT,
    AvgSessionDurationMinutes INT,
    PlayerLevel INT,
    AchievementsUnlocked INT,
    EngagementLevel STRING
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 's3://your-game-analytics-bucket/raw/'
TBLPROPERTIES ("skip.header.line.count"="1");