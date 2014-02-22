function [norm_frame] = RGBNormalize(frame)
% Normalize the RGB Image
    
    % Normalize RED
%     channel = double(frame(:, :, 1));
%     m = mean(channel(:));
%     s = std(channel(:));
%     red = (channel - m) / s;
    
    % Normalize GREEN
%     channel = double(frame(:, :, 2));
%     m = mean(channel(:));
%     s = std(channel(:));
%     green = (channel - m) / s;
    
    % Normalize BLUE
%     channel = double(frame(:, :, 3));
%     m = mean(channel(:));
%     s = std(channel(:));
%     blue = (channel - m) / s;
    
    % Concatenate the images
    %frame = cat(3, red, green, blue);
    
    norm_frame = double(frame); %double(zeros(size(frame)));

%     for i = 1 : size(frame,1)
%        for j = 1 : size(frame,2)
%           red = norm_frame(i,j,1);
%           green = norm_frame(i,j,2);
%           blue = norm_frame(i,j,3);
%           sum = red + green + blue;
%           
%           norm_frame(i,j,1) = red/sum;
%           norm_frame(i,j,2) = green/sum;
%           norm_frame(i,j,3) = blue/sum;
%           
%        end
%     end

    R = norm_frame(:,:,1);
    G = norm_frame(:,:,2);
    B = norm_frame(:,:,3);
    
    sum = R+G+B;

    norm_frame(:,:,1) = R./sum;
    norm_frame(:,:,2) = G./sum;
    norm_frame(:,:,3) = B./sum;
    
end

