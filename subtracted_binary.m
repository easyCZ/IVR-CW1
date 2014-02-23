
file_dir = 'GOPR0002/';
filenames = dir([file_dir '*.jpg']);

frame = imread([file_dir filenames(1).name]);
new_frame = zeros(size(frame,1), size(frame,2));
figure(1); h1 = imshow(new_frame);

%create bg
bg = RGBNormalize(BGSub(25, file_dir));

threshold = 0.075;

for k = 25 : size(filenames, 1)
   
    disp([file_dir filenames(k).name]);
    frame = RGBNormalize( imread([file_dir filenames(k).name]) );
    frame = frame - bg;
    frame(frame < threshold) = 0;
    frame(frame >= threshold) = 1;

    new_frame = sum(frame, 3);
    new_frame(new_frame<1) = 0;
    new_frame(new_frame>=1) = 1;
    
    new_frame = bwmorph(new_frame,'erode',1);

    % You can apply a filter to the image. Currently it's Sobel.
    new_frame = kernel_blur(new_frame);  
    
    set(h1, 'CData', new_frame);
    drawnow('expose');
    
end