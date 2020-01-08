function [img_out] = func_prob3( image_left, image_right, disparity_range, win_size)
% By Lukas Gabert
% For 3D computer vision

[ywidth,xwidth,~] = size(image_left);
img_left = rgb2gray(image_left);
img_right = rgb2gray(image_right);
img_out = zeros(ywidth,xwidth,1,'uint8');

scaling = round(255/disparity_range);
extend = int16(round((win_size-1)/2));
    
for y = 1:ywidth
    for x = 1:xwidth
        bestDisparity = 0;
        bestNCC = 0;
        for disp = 1:disparity_range
            if (y-extend < 1) || (y+extend > ywidth) ||...
                    (x-extend-disp < 1) || (x+extend > xwidth)
                currNCC = 0;
            else
                patch1 = img_left(y-extend:y+extend,x-extend:x+extend);
                patch2 = img_right(y-extend:y+extend,...
                    x-disp-extend:x-disp+extend);
                currNCC = NCC(patch1,patch2);
            end
            if currNCC > bestNCC
                bestNCC = currNCC;
                bestDisparity = disp;
            end
        end
        img_out(y,x) = bestDisparity*scaling;
    end
end

imshow(img_out)

end

