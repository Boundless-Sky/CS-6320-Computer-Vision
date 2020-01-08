function [out] = makes_plot(image, lineSet, minPixToRemove, on)
% By Lukas Gabert
% For plotting lines in matlab
    figure()
    %import image
    BW = rgb2gray(image);
    %get edges of image
    edges2 = edge(BW, 'canny');
    edges = bwareaopen(edges2, minPixToRemove);
    % edges = edges2;
    ywidth = size(edges(:,1));
    xwidth = size(edges(:,2));
    xwidth = xwidth(1);
    ywidth = ywidth(1);
    numLines = size(lineSet);
    numLines = numLines(1);
    
    
    imshow(edges)
    hold on
    for i = 1 : numLines
        a = cell2mat(lineSet(i));
        if isempty(a)
            break
        end
        y = (a(:,2));
        x = a(:,1);
        plot(x,y, '-','Linewidth',3)
    end
    if on
        plot(xwidth-70, 70, 'g.', 'MarkerSize', 70)
    else
        plot(xwidth-70, 70, 'r.', 'MarkerSize', 70)
    end
    hold off
end

