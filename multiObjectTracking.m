function multiObjectTracking()

    % Needs to be there in order to avoid some Matlab bug.
    ones(10)*ones(10);

    paused = java.util.ArrayList();
    % Create system objects used for reading video, detecting moving objects,
    % and displaying the results.
    obj = setupSystemObjects();

    tracks = initializeTracks(); % Create an empty array of tracks.

    nextId = 1; % ID of the next track

    file_dir = 'GOPR0004/'; %put here one of the folder locations with images;
    filenames = dir([file_dir '*.jpg']);

    frame = imread([file_dir filenames(1).name]);

    % Detect moving objects, and track them across video frames.
    for k = 1 : size (filenames, 1)
        frame = imread([file_dir filenames(k).name]);
        [centroids, bboxes, mask, majora, minora, eccentricities, perimeters] = detectObjects(frame);


        predictNewLocationsOfTracks();
        [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment();

        updateAssignedTracks();
        updateUnassignedTracks();
        deleteLostTracks();
        createNewTracks();

        displayTrackingResults();
    end

    function obj = setupSystemObjects()
        % Initialize Video I/O
        % Create objects for reading a video from a file, drawing the tracked
        % objects in each frame, and playing the video.

        % Create two video players, one to display the video,
        % and one to display the foreground mask.
        obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 520]);
        obj.maskPlayer = vision.VideoPlayer('Position', [740, 400, 700, 520]);

        % Create system objects for foreground detection and blob analysis

        % The foreground detector is used to segment moving objects from
        % the background. It outputs a binary mask, where the pixel value
        % of 1 corresponds to the foreground and the value of 0 corresponds
        % to the background.

        obj.detector = vision.ForegroundDetector('NumGaussians', 2, ...
            'NumTrainingFrames', 25, 'MinimumBackgroundRatio', 0.8, ...
            'InitialVariance', 25*25, 'AdaptLearningRate', true, 'LearningRate', 0.0001);

        % Connected groups of foreground pixels are likely to correspond to moving
        % objects.  The blob analysis system object is used to find such groups
        % (called 'blobs' or 'connected components'), and compute their
        % characteristics, such as area, centroid, and the bounding box.

        obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MajorAxisLengthOutputPort', true, 'MinorAxisLengthOutputPort', true, ...
            'EccentricityOutputPort', true, 'PerimeterOutputPort', true, ...
            'MinimumBlobArea', 200);



    end

    function tracks = initializeTracks()
        % create an empty array of tracks
        tracks = struct(...
            'id', {}, ...
            'bbox', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {}, ...
            'stack', {}, ...
            'max_x', {}, ...
            'max_y', {}, ...
            'last_x', {}, ...
            'last_y', {}, ...
            'should_pause', {}, ...
            'stop_pausing', {})


    end

    function [centroids, bboxes, mask, majora, minora, eccentricities, perimeters] = detectObjects(frame)

        % Detect foreground.
        mask = obj.detector.step(frame);

        % Apply morphological operations to remove noise and fill in holes.

        % mask = imopen(mask, strel('rectangle', [3,3]));
        % mask = imclose(mask, strel('rectangle', [15, 15]));

        % mask = imopen(mask, strel('octagon', 3));
        % mask = imclose(mask, strel('octagon', 9));

        % mask = imfill(mask, 'holes');

        % Perform blob analysis to find connected components.
        [area, centroids, bboxes, majora, minora, eccentricities, perimeters] = obj.blobAnalyser.step(mask);
    end

    function predictNewLocationsOfTracks()
        for i = 1:length(tracks)
            bbox = tracks(i).bbox;

            % Predict the current location of the track.
            predictedCentroid = predict(tracks(i).kalmanFilter);

            % Shift the bounding box so that its center is at
            % the predicted location.
            predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
            tracks(i).bbox = [predictedCentroid, bbox(3:4)];
        end
    end

    function [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment()

        nTracks = length(tracks);
        nDetections = size(centroids, 1);

        % Compute the cost of assigning each detection to each track.
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
        end

        % Solve the assignment problem.
        costOfNonAssignment = 20;
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, costOfNonAssignment);
    end

    function updateAssignedTracks()
        numAssignedTracks = size(assignments, 1);
        balls = isBall(eccentricities, perimeters, bboxes, majora, minora);

        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            centroid = centroids(detectionIdx, :);
            bbox = bboxes(detectionIdx, :);

            % Correct the estimate of the object's location
            % using the new detection.
            correct(tracks(trackIdx).kalmanFilter, centroid);

            % Replace predicted bounding box with detected
            % bounding box.
            tracks(trackIdx).bbox = bbox;

            % Update track's age.
            tracks(trackIdx).age = tracks(trackIdx).age + 1;

            % Update track stats
            x = bbox(1) + floor(bbox(3) / 2);
            y = bbox(2);

            if tracks(trackIdx).last_y ~= Inf && tracks(trackIdx).last_y ~= -1
                deltaY = tracks(trackIdx).last_y - y;
            else
                deltaY = 1;
            end

            % Re-assign values if current max
            if y < tracks(trackIdx).max_y
                tracks(trackIdx).max_x = x;
                tracks(trackIdx).max_y = y;
            end
            % Draw only for positive values
            if tracks(trackIdx).max_y > 0 && tracks(trackIdx).max_y < Inf
                text = strcat('x:', int2str(tracks(trackIdx).max_x), ' y:', int2str(tracks(trackIdx).max_y));
                frame = insertMarker(frame, [tracks(trackIdx).max_x, tracks(trackIdx).max_y], 'Size', 15);
                frame = insertText(frame, [tracks(trackIdx).max_x + 1, tracks(trackIdx).max_y - 23], text, 'FontSize', 13, 'BoxColor', 'red', 'BoxOpacity', 0.4);
            end

            % Update last frame cache
            tracks(trackIdx).last_x = x;
            tracks(trackIdx).last_y = y;

            % tracks(trackIdx).max_x = tracks(trackIdx).max_x;
            % tracks(trackIdx).max_y = tracks(trackIdx).max_y;

            % Update visibility.
            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
            tracks(trackIdx).consecutiveInvisibleCount = 0;

            tracks(trackIdx).stack.push(balls.get(i - 1));

            tracks(trackIdx).should_pause = false;
            if deltaY < 0 && ~paused.contains(trackIdx)
                if getBallProbability(tracks(trackIdx).stack) > 0.9
                tracks(trackIdx).should_pause = true;
                end
            end
        end
    end

    function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;
            tracks(ind).consecutiveInvisibleCount = ...
                tracks(ind).consecutiveInvisibleCount + 1;
        end
    end

    function deleteLostTracks()
        if isempty(tracks)
            return;
        end

        invisibleForTooLong = 4;
        ageThreshold = 8;

        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;

        % Find the indices of 'lost' tracks.
        lostInds = (ages < ageThreshold & visibility < 0.6) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;

        % Delete lost tracks.
        tracks = tracks(~lostInds);
    end

    function createNewTracks()
        centroids = centroids(unassignedDetections, :);
        bboxes = bboxes(unassignedDetections, :);

        for i = 1:size(centroids, 1)

            centroid = centroids(i,:);
            bbox = bboxes(i, :);

            % Create a Kalman filter object.
            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, [200, 50], [100, 25], 100);

            % Create a new track.
            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0, ...
                'stack', java.util.Stack(), ...
                'max_x', -1, ...
                'max_y', Inf, ...
                'last_x', -1, ...
                'last_y', Inf, ...
                'should_pause', false, ...
                'stop_pausing', false);

            % Add it to the array of tracks.
            tracks(end + 1) = newTrack;

            % Increment the next id.
            nextId = nextId + 1;
        end
    end

    function displayTrackingResults()
        % Convert the frame and the mask to uint8 RGB.
        frame = im2uint8(frame);
        mask = uint8(repmat(mask, [1, 1, 3])) .* 255;

        minVisibleCount = 8;
        if ~isempty(tracks)

            % Noisy detections tend to result in short-lived tracks.
            % Only display tracks that have been visible for more than
            % a minimum number of frames.
            reliableTrackInds = ...
                [tracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = tracks(reliableTrackInds);

            % Display the objects. If an object has not been detected
            % in this frame, display its predicted bounding box.
            if ~isempty(reliableTracks)

                % Get bounding boxes.
                bboxes = cat(1, reliableTracks.bbox);

                % Get ids.
                ids = int32([reliableTracks(:).id]);

                % Create labels for objects indicating the ones for
                % which we display the predicted rather than the actual
                % location.
                labels = cellstr(int2str(ids'));
                predictedTrackInds = ...
                    [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {' predicted'};
                labels = strcat(labels, isPredicted);

                % Draw the objects on the frame.
                frame = insertObjectAnnotation(frame, 'rectangle', ...
                    bboxes, labels);

                % Draw the objects on the mask.
                mask = insertObjectAnnotation(mask, 'rectangle', ...
                    bboxes, labels);

                for i = 1 : size(reliableTracks, 2)
                    if reliableTracks(i).should_pause && ~paused.contains(reliableTracks(i).id)
                        pause(3);
                        paused.add(reliableTracks(i).id);
                    end
                end



            else
                stopPausing = false;
                % ballStack = java.util.Stack();
            end


        end

        % Display the mask and the frame.
        obj.maskPlayer.step(mask);
        obj.videoPlayer.step(frame);

    end

    function prob = getBallProbability(ballStack)
        % Calculate the probability that the image has balls. Yay!

        stackSize = ballStack.size();

        count = 0;

        while ~ballStack.isEmpty()
            val = ballStack.pop();
            if val
                count = count + 1;
            end
        end

        prob = count / stackSize;

        if stackSize < 8
            prob = 0;
        end
    end

    function balls = isBall(eccentricities, perimeters, bboxes, majors, minors)

        balls = java.util.ArrayList()

        for i = 1 : size(eccentricities)
            if eccentricities(i) < 0.8
                estimatedRadius = (majors(i) + minors(i))/2;
                estimatedPerimeter = pi * estimatedRadius;

                ratio = perimeters(i)/estimatedPerimeter;
                balls.add(ratio <1.2);
            else
                balls.add(false);
            end
        end
    end

end