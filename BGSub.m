function [averaged] = BGSub(frame_count, file_dir)
    % Create background subtraction image

    filenames = dir([file_dir '*.jpg']);
    
    %Temporary list
    list = zeros(480, 640, 3, frame_count);
    
    for k = 1: frame_count
        frame = imread([file_dir filenames(k).name]);
        list(:, :, :, k) = frame;
    end    
    averaged = uint8(median(list, 4));

end
