
FILE_DIR = 'GOPR0002/';
filenames = dir([file_dir '*.jpg']);

%create bg
bg = BGSub(25, FILE_DIR);

for k = 25: size(filenames, 1)
   
    disp([file_dir filenames(k).name]);
    frame = imread([file_dir filenames(k).name]) - bg;
    set(h1, 'CData', frame);
    drawnow('expose');
    
end