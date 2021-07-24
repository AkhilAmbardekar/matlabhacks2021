% read all data
allData = readtable("crashes_small.xlsx");

% extract output
severity = allData.Severity;

% extract inputs
distance = allData.Distance_mi_';
temperature = allData.Temperature_F_';
humidity = allData.Humidity___';
visibility = allData.Visibility_mi_';
windSpeed = allData.Wind_Speed_mph_';
crossing = allData.Crossing';
crossing = double(crossing);
stop = allData.Stop';
trafficSignal = allData.Traffic_Signal';

% ordinal encoding for weather
weatherCond = allData.Weather_Condition';
weatherCond = categorical(weatherCond);
weatherCondOrd = grp2idx(weatherCond)'; % use unique to find how it has been transcribed

dayNight = allData.Sunrise_Sunset';
dayNight = categorical(dayNight);
dayNightOrd = grp2idx(dayNight);
dayNightOrd = dayNightOrd - ones(1, 50000);

inputs = [distance; temperature; humidity; visibility; windSpeed; ...
    crossing; stop; trafficSignal; weatherCondOrd; dayNightOrd];
    
outputs = [severity];