% Played around with different kinds of filters.
% The Sobel filter is supposed to do edge detection (and it looks
% quite cool if you're applying it on the raw images), but on the
% masked ones is only detects the bottom edges. I don't now why it's 
% doing that. Still, it might be useful for something.

function [blurred_frame] = kernel_blur(frame)

	% sigma = 1.0;
	% h = fspecial('guassian', size(frame));
	% blurred_frame = imfilter(frame, h, sigma);

	% h = fspecial('disk', 10);
	% blurred_frame = imfilter(frame, h, 'replicate');

	h = fspecial('sobel');
	blurred_frame = imfilter(frame, h, 'replicate');

end