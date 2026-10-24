function generate_cyclone_footprints(input_csv, output_csv, er11_dir)
%GENERATE_CYCLONE_FOOTPRINTS Evaluate ER11 profiles for a chunk of centers.
%
% The input table contains global_id, longitude, latitude, maximum_wind_ms
% and radius_m. The output contains one outer wind radius for every 2 m/s
% contour strictly above 18 m/s, matching the levels plotted by the legacy
% draw_cyclone.py/draw_cascade.py workflow.

if nargin ~= 3
    error('Expected input CSV, output CSV and ER11 directory.');
end
if ~isfile(input_csv)
    error('Cyclone-center input does not exist: %s', input_csv);
end
if ~isfolder(er11_dir)
    error('ER11 directory does not exist: %s', er11_dir);
end

addpath(er11_dir);
cleanup = onCleanup(@() rmpath(er11_dir)); %#ok<NASGU>
centers = readtable(input_csv);
required = {'global_id', 'longitude', 'latitude', 'maximum_wind_ms', 'radius_m'};
if ~all(ismember(required, centers.Properties.VariableNames))
    error('Cyclone-center input is missing required columns.');
end

out_global_id = zeros(0, 1);
out_longitude = zeros(0, 1);
out_latitude = zeros(0, 1);
out_wind_speed = zeros(0, 1);
out_radius = zeros(0, 1);

for row = 1:height(centers)
    maximum_wind = centers.maximum_wind_ms(row);
    maximum_radius = centers.radius_m(row);
    highest_level = floor(maximum_wind / 2) * 2;
    levels = (20:2:highest_level)';
    if isempty(levels) || maximum_radius <= 0
        continue;
    end

    [radii, winds] = ER11E04_nondim_rmaxinput( ...
        maximum_wind, maximum_radius, 5e-5, 0, 1.5e-3, ...
        2 / 1000, 0, 1.0, 0, 0.15);
    radii = radii(:);
    winds = winds(:);
    outer_start = find(radii >= maximum_radius, 1, 'first');
    if isempty(outer_start)
        [~, outer_start] = min(abs(radii - maximum_radius));
    end
    outer_radius = radii(outer_start:end);
    outer_wind = winds(outer_start:end);
    finite = isfinite(outer_radius) & isfinite(outer_wind);
    outer_radius = outer_radius(finite);
    outer_wind = outer_wind(finite);
    if numel(outer_radius) < 2
        continue;
    end

    [wind_axis, unique_index] = unique(-outer_wind);
    radius_axis = outer_radius(unique_index);
    if numel(wind_axis) < 2
        continue;
    end
    radius_at_level = interp1( ...
        wind_axis, radius_axis, -levels, 'linear', 'extrap');
    keep = isfinite(radius_at_level) & radius_at_level >= 0;
    levels = levels(keep);
    radius_at_level = radius_at_level(keep);
    count = numel(levels);
    if count == 0
        continue;
    end

    out_global_id = [out_global_id; repmat(centers.global_id(row), count, 1)]; %#ok<AGROW>
    out_longitude = [out_longitude; repmat(centers.longitude(row), count, 1)]; %#ok<AGROW>
    out_latitude = [out_latitude; repmat(centers.latitude(row), count, 1)]; %#ok<AGROW>
    out_wind_speed = [out_wind_speed; levels]; %#ok<AGROW>
    out_radius = [out_radius; radius_at_level]; %#ok<AGROW>
end

footprints = table( ...
    out_global_id, out_longitude, out_latitude, out_wind_speed, out_radius, ...
    'VariableNames', ...
    {'global_id', 'longitude', 'latitude', 'wind_speed_ms', 'radius_m'});
writetable(footprints, output_csv);
end
