file_dir = 'GOPR0002/';
filenames = dir([file_dir '*.jpg']);

frame = imread([file_dir filenames(1).name]);
figure(1); h1 = imshow(frame);

obj.detector = vision.ForegroundDetector('NumGaussians', 3, 'NumTrainingFrames', 25, 'MinimumBackgroundRatio', 0.5);

for k = 1 : size(filenames, 1)
    frame = imread([file_dir filenames(k).name]);
    mask = obj.detector.step(frame);
    
    set(h1, 'CData', mask);
    drawnow('expose');
end