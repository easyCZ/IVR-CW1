
file_dir = 'GOPR0002/';
filenames = dir([file_dir '*.jpg']);

frame = imread([file_dir filenames(1).name]);
figure(1); h1 = imshow(frame);

%create bg
bg = RGBNormalize(BGSub(25, file_dir));

threshold = 0.075;

for k = 25 : size(filenames, 1)
   
    disp([file_dir filenames(k).name]);
    frame = RGBNormalize( imread([file_dir filenames(k).name]) );
    frame = frame - bg;
    frame(frame < threshold) = 0;
    frame(frame >= threshold) = 1;
    
    frame(:,:) = bwmorph(frame(:,:),'erode',1);
    
    set(h1, 'CData', frame);
    drawnow('expose');
    
end