function [img_out] = func_prob2( image, r_min, r_max, g_min, g_max, b_min, b_max )
% By Lukas Gabert
% For 3D computer vision

[ywidth,xwidth,zwidth] = size(image);
img_out = zeros(ywidth,xwidth,zwidth,'uint8');
    
for i = 1:ywidth
    for j = 1:xwidth
        if (image(i,j,1) > r_min) && (image(i,j,1) < r_max) &&...
                (image(i,j,2) > g_min) && (image(i,j,2) < g_max) &&...
                (image(i,j,3) > b_min) && (image(i,j,3) < b_max)
            img_out(i,j,:) = [255,255,255];
        end
    end
end

imshow(img_out)

end

