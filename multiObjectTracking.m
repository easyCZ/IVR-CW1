function multiObjectTracking()

    % Needs to be there in order to avoid some Matlab bug.
    ones(10)*ones(10);

    % Display video
    obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 520]);

    % Backgrond model and object detector
    obj.detector = vision.ForegroundDetector('NumGaussians', 2, ...
            'NumTrainingFrames', 25, 'MinimumBackgroundRatio', 0.8, ...
            'InitialVariance', 25*25, 'AdaptLearningRate', true, 'LearningRate', 0.0001);

    % Blob analysis and recognition
    obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MajorAxisLengthOutputPort', true, 'MinorAxisLengthOutputPort', true, ...
            'EccentricityOutputPort', true, 'PerimeterOutputPort', true, ...
            'MinimumBlobArea', 200);

    % Global state variables
    paused = java.util.ArrayList();

     % Create an empty array of tracks.
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
        'stop_pausing', {}, ...
        'track_xs', {}, ...
        'track_ys', {});

    % ID of the next track
    nextId = 1;

    % Directory to take images from
    file_dir = 'GOPR0004/';
    filenames = dir([file_dir '*.jpg']);

    % Detect moving objects, and track them across video frames.
    % Main loop of the program
    for k = 1 : size (filenames, 1)
        % Load an image
        frame = imread([file_dir filenames(k).name]);

        % Find objects on the image
        [centroids, bboxes, mask, majora, minora, eccentricities, perimeters] = detectObjects(frame);

        % Attempt to predict the location of objects
        predictNewLocationsOfTracks();

        % Match detected objects to existing objects
        [assignments, unassignedTracks, unassignedDetections] = detectionToTrackAssignment();

        % Update informations for each track - maximums and ball/not ball classification
        updateAssignedTracks();

        % Increment age of unassigned tracks to be considered for deletion later
        updateUnassignedTracks();

        % Delete the tracks that are too old or not visible
        deleteLostTracks();

        % Create new objects from unassigned tracks
        createNewTracks();

        % Draw GUI
        displayTrackingResults();
    end

    function [centroids, bboxes, mask, majora, minora, eccentricities, perimeters] = detectObjects(frame)
        % Detect foreground.
        mask = obj.detector.step(frame);

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

            % Update visibility.
            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
            tracks(trackIdx).consecutiveInvisibleCount = 0;

            tracks(trackIdx).stack.push(balls.get(i - 1));

            % Determine if we should be pausing for this object
            tracks(trackIdx).should_pause = false;
            if deltaY < 0 && ~paused.contains(trackIdx)
                % Only pause if the probabilty of being a ball given the past is past the threshold
                if getBallProbability(tracks(trackIdx).stack) > 0.8
                    tracks(trackIdx).should_pause = true;
                end
            end

            % Add current location to the history of location
            tracks(trackIdx).track_xs.add(x);
            tracks(trackIdx).track_ys.add(y);

            % Check how many track points there are in the object
            numTracks = tracks(trackIdx).track_ys.size();
            points = zeros(numTracks, 2);

            % Precompute a [numTracks x 2] matrix to draw markers
            for i = 1 : tracks(trackIdx).track_ys.size()
                points(i, 1) = tracks(trackIdx).track_xs.get(i-1);
                points(i, 2) = tracks(trackIdx).track_ys.get(i-1);
            end

            % Draw points
            frame = insertMarker(frame, points, 'o', 'Size', 1);
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
                'stop_pausing', false, ...
                'track_xs', java.util.ArrayList(), ...
                'track_ys', java.util.ArrayList());

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
%                 mask = insertObjectAnnotation(mask, 'rectangle', ...
%                     bboxes, labels);

                for i = 1 : size(reliableTracks, 2)
                    if reliableTracks(i).should_pause && ~paused.contains(reliableTracks(i).id)
                        pause(3);
                        paused.add(reliableTracks(i).id);
                    end
                end
            end


        end

        % Display the mask and the frame.
%         obj.maskPlayer.step(mask);
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
        balls = java.util.ArrayList();

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