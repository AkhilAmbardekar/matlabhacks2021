% read all data
allData = readtable("crashes_small.xlsx");

% extract output
severity = allData.Severity;

% extract inputs
startTime = allData.Start_Time;
endTime = allData.End_Time;
duration = endTime - startTime;
duration = duration';
duration = datenum(duration);

temperature = allData.Temperature_F_';
humidity = allData.Humidity___';
visibility = allData.Visibility_mi_';
windSpeed = allData.Wind_Speed_mph_';
crossing = allData.Crossing';
crossing = double(crossing);
mStop = allData.Stop';
mStop = double(mStop);
trafficSignal = allData.Traffic_Signal';
trafficSignal = double(trafficSignal);

% ordinal encoding for weather
weatherCond = allData.Weather_Condition';
weatherCond = categorical(weatherCond);
weatherCondOrd = grp2idx(weatherCond)'; % use unique to find how it has been transcribed

dayNight = allData.Sunrise_Sunset';
dayNight = categorical(dayNight);
dayNightOrd = grp2idx(dayNight)';
dayNightOrd = dayNightOrd - ones(1, 50000);

Predictors = [duration; temperature; humidity; visibility; windSpeed; ...
    crossing; mStop; trafficSignal; weatherCondOrd; dayNightOrd]';
    
Response = categorical(severity);

netData = table(Predictors, Response);
