function [frame] = RGBNormalize(frame)
% Normalize the RGB Image
    
    % Normalize RED
    channel = double(frame(:, :, 1));
    m = mean(channel(:));
    s = std(channel(:));
    red = (channel - m) / s;
    
    % Normalize GREEN
    channel = double(frame(:, :, 2));
    m = mean(channel(:));
    s = std(channel(:));
    green = (channel - m) / s;
    
    % Normalize BLUE
    channel = double(frame(:, :, 3));
    m = mean(channel(:));
    s = std(channel(:));
    blue = (channel - m) / s;
    
    % Concatenate the images
    frame = cat(3, red, green, blue);

end

