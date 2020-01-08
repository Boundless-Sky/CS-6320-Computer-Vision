function [newLineSet, stairLineSet, closeToStairs] = line_identify_coord(lineSet, err_thresh, minlines, thresh, diff_thresh)
% By Justin Ngo and Lukas Gabert
% Find the primary coordinate system by orgainizing number 
% of similar slopes and taking the top three
% New Line Set: Set of orthogonal lines (x,y,z)
%     x is horzontal, y is depth, z is height
% Ortho Lines: 3 lines (make more probably) (i, 3)
%              1 is xrun, 2 is yrun, 3 is length
orthoLines = zeros(50,200);
orthoPlace = 1;
for i = 1:length(lineSet)
    currLine = cell2mat(lineSet(i,:));
    pt1 = currLine(1,:);
    pt2 = currLine(2,:);
    length_ = norm(pt1-pt2);
    xrun = (pt2(1)-pt1(1))/length_;
    yrun = (pt2(2)-pt1(2))/length_;
%     if xrun and yrun in set of xs, then blah blah
%     else create new set thingy
    needsHome = 1;
    for j = 1:size(orthoLines)
        check_line = orthoLines(j,1:2);
       if xrun <= check_line(1) + err_thresh && xrun >= check_line(1) - err_thresh...
       && yrun <= check_line(2) + err_thresh && yrun >= check_line(2) - err_thresh 
           orthoLines(j,3) = orthoLines(j,3) + length_;
           orthoLines(j,4) = orthoLines(j,4) + 1;
           orthoLines(j,orthoLines(j,4)) = i;
           needsHome = 0;
       end
    end
    if needsHome
        orthoLines(orthoPlace,1:3) = [xrun, yrun, length_];
        orthoLines(orthoPlace,4) = 5;
        orthoLines(orthoPlace,5) = i;
        orthoPlace = orthoPlace + 1;
    end
end

%Arrange OrthoLines by Length, take top 3
[~, I] = sort(orthoLines,'descend');
a = I(1,3);
c = orthoLines(a,5:end);
c = c(c ~= 0);
b = I(2,3);
d = orthoLines(b,5:end);
d = d(d ~= 0);
c = [c d];
newLineSet = lineSet(c);

%organize lines by y dimensions with in a threshold so that you can start
%drawing single lines
singlelines = cell2mat(newLineSet);
% threshold = 50; %pixels
%format is x and y 
%find all lines within a y range just looking at first point
point1 = singlelines(1:2:end,:);
point2 = singlelines(2:2:end,:);
[p, q] = sort(point1(:,2));
[r, s] = sort(point2(:,2));

% just see if lines are in same "horizon"
numlines = 0;
currentvalue = 0;
currentvalue_new = 0;
cellOfIndices = {};
cellCount = 1;
indices = [];
for ii = 1:length(p)
    currentvalue_new = p(ii);
    if abs(currentvalue_new - currentvalue) < thresh
        numlines = numlines + 1;
        indices = [indices, ii];
    else
        if numlines >= minlines
            cellOfIndices{cellCount,1} = indices;
            cellCount = cellCount + 1;
        end
        indices = [ii];
        numlines = 1;
    end
    currentvalue = currentvalue_new;
end

%create big single lines using min max 
%find min max of y...
stairLineSet={};
saveymin = [];
for jj = 1:length(cellOfIndices)
    yarray = cell2mat(cellOfIndices(jj));
    yvalues = [point1(q(yarray),2); point2(q(yarray),2)];
    xvalues = [point1(q(yarray),1); point2(q(yarray),1)];
    [xsorted, xindices] = sort(xvalues);
    ymin = yvalues(xindices(1));
    ymax = yvalues(xindices(end));
    if xsorted(end) - xsorted(1) > 1000
        stairLineSet{jj,1} = [xsorted(1) ymin; xsorted(end), ymax];
        saveymin = [ saveymin , ymin];
    else
        continue
    end
end
%count number of lines and see if there are x amount awway from each other
%then you call as stairs
closeToStairs = 0;
spacedLineCount = 0;
if length(saveymin) > 2
    for kk = 1:length(saveymin)-1
        diff = abs(saveymin(kk + 1) - saveymin(kk));
        if diff > diff_thresh && diff < (diff_thresh + 50)
            spacedLineCount = spacedLineCount + 1;
        end
    end
end
if spacedLineCount > 1
    closeToStairs = 1;
end
%if it detects 3 lines then its a stair
end

