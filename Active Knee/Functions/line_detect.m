% By Lukas Gabert
% 3D Computer Vision Spring 2018

function out = line_detect(image, numiters, maxPairDist, minPtDist, minPixNum, minPixToRemove)
    %Create the set of lines
    lineSet = {};

    %import image
    BW = rgb2gray(image);
    %get edges of image
    edges2 = edge(BW, 'canny');
    % Remove objects containing fewer than minPixToRemove pixels
    edges = bwareaopen(edges2, minPixToRemove);
    % edges = edges2;
    xwidth = size(edges(1,:));
    xwidth = xwidth(2);
    ywidth = size(edges(:,1));
    ywidth = ywidth(1);
    [y,x] = find(edges == 1);
    % y = ywidth - y;
    edgePixelSet = [x,y];
   
    
    % imshow(edges)
    
    

    
%run for set number of iterations
for i = 1:numiters
    
    % sample randomly from edge pixels
    edgeSize = numel(edgePixelSet(:,1));
    randPixel = randperm(edgeSize, 1);
    pixelx = x(randPixel,:);
    pixely = y(randPixel,:);
    
    % Sample another pixel, close to first
    if pixelx > maxPairDist+minPtDist &&...
            pixelx < xwidth - maxPairDist-minPtDist &&...
            pixely > maxPairDist+minPtDist &&...
            pixely < ywidth - maxPairDist-minPtDist
%     ynew = y(pixely-maxPairDist:pixely+maxPairDist);
%     xnew = x(pixelx-maxPairDist:pixelx+maxPairDist);
      ynew1 =  y(y(:,1)>pixely-maxPairDist & y(:,1)<pixely+maxPairDist &...
          x(:,1) >= pixelx-maxPairDist & x(:,1) < pixelx+maxPairDist);
      xnew1 =  x(y(:,1)>pixely-maxPairDist & y(:,1)<pixely+maxPairDist &...
          x(:,1) >= pixelx-maxPairDist & x(:,1) < pixelx+maxPairDist);
    else
        continue
    end
%     xnew1 = xnew + pixelx - maxPairDist;
%     ynew1 = ywidth - (ynew + pixely - maxPairDist);
    newSize = numel(xnew1);
    randPixNew = randperm(newSize, 1);
    pixelxNew = xnew1(randPixNew,:);
    pixelyNew = ynew1(randPixNew,:);
    pt1 = [pixelx, pixely];
    pt2 = [pixelxNew, pixelyNew];
    
    % plot(xnew1,ynew1,'g.')
    % plot([pt1(1),pt2(1)],[pt1(2),pt2(2)],'r','LineWidth',2)
    
    
    %Calculate distance to line between points
    numInPath = 0;
    % ptsInLine = zeros(sqrt(xwidth*ywidth),2);
    ptsInLine = [];
    j = 1;
    m = pt1 - pt2;
    m = m./norm(m);
    edges2 = edges;
    endpt1 = pt2;
    endpt2 = pt2;
    %start from pt2
    ptx = pt2(1); pty = pt2(2);
    while true
        d = int16(floor(pty-minPtDist));
        e = int16(ceil(pty+minPtDist));
        f = int16(floor(ptx-minPtDist));
        g = int16(ceil(ptx+minPtDist));
        if (d < 1) || (f < 1) || (e > ywidth) || (g > xwidth)
            break
        end
        [y1,x1] = find(edges2(d:e,f:g) == 1);
        if isempty(y1)
            break
        end
        y1 = y1 + double(d);
        x1 = x1 + double(f);
        ptsInLine = [ptsInLine; [x1,y1]];
        endpt1 = [x1(1),y1(1)];
        ptx = ptx+m(1);
        pty = pty+m(2);
        iter = 1;
        while iter < length(y1)
            edges2(y1(iter),x1(iter)) = 0;
            iter = iter + 1;
        end
        numInPath = numInPath + 1;
    end
    %extend toward pt1
    while true
        d = int16(floor(pty-minPtDist));
        e = int16(ceil(pty+minPtDist));
        f = int16(floor(ptx-minPtDist));
        g = int16(ceil(ptx+minPtDist));
        if (d < 1) || (f < 1) || (e > ywidth) || (g > xwidth)
            break
        end
        [y1,x1] = find(edges2(d:e,f:g));
        if isempty(y1)
            break
        end
        y1 = y1 + double(d);
        x1 = x1 + double(f);
        ptsInLine = [ptsInLine; [x1,y1]];
        endpt2 = [x1(1),y1(1)];
        ptx = ptx-m(1);
        pty = pty-m(2);
        iter = 1;
        while iter < length(y1)
            edges2(y1(iter),x1(iter)) = 0;
            iter = iter + 1;
        end
        numInPath = numInPath + 1;
    end
    %count num pts
    %If greater than a threshold,delete edge pixels from the set
    
    if  numInPath > minPixNum
        edges(ptsInLine(:,2),ptsInLine(:,1)) = 0;
        lineSet = [lineSet;[endpt1;endpt2]];
    end
end

out = lineSet;

end