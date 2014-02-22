classdef Feed
    %FEED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        FrameNumber = 0;
        FileNames = [];
    end
    
    methods
        function Feed(folder)
            FileNames = dir([folder '*.jpg']);
        end
        
        function nextFrame = next()
            
        end
    end
    
end

