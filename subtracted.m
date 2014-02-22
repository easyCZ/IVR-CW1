
FILE_DIR = 'GOPR0002/';
filenames = dir([file_dir '*.jpg']);

figure(1); h1 = imshow(frame);

%create bg
% bg = BGSub(25, FILE_DIR);
bg = RGBNormalize(BGSub(25, FILE_DIR));

for k = 25: size(filenames, 1)
   
    disp([file_dir filenames(k).name]);
%     frame = RGBNormalize(( imread([file_dir filenames(k).name])-bg));
    frame = RGBNormalize( imread([file_dir filenames(k).name]) );
    frame = frame - bg;
    frame(frame<0.075) = 0;
    frame(frame>=0.075) = 1;
    
    set(h1, 'CData', frame);
    drawnow('expose');
    
end