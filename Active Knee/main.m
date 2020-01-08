% By Justin Ngo and Lukas Gabert
% For 3D Computer Vision, FINAL PROJECT
% Spring 2018

% clc
clearvars
close all

%Create paths for all folders
MyPath = userpath;
MyDir = MyPath(1:strfind(MyPath,';')-1);
MyWorkDir = genpath(MyDir);
addpath(MyWorkDir, 'Outputs');
addpath(MyWorkDir, 'Inputs');
addpath(MyWorkDir, 'Functions');
addpath(MyWorkDir, 'TOOLBOX_calib');
addpath(MyWorkDir, 'siftDemoV4');

%%  Run the function for problem 1

tic
a = 'IMG_';
c = '.jpg';
d = '/Outputs/IMG_';
e = '_Preprocessed.png';
f = '_Medprocessed.png';
g = '_Processed.png';
for b = 8907:8907
    close all
    Im1_1 = imread(strcat(a,num2str(b),c));
    lineSet_total = line_detect(Im1_1, 10000, 350, 2, 70, 200);
    [lineSet_coord, stairLines, on] = line_identify_coord(lineSet_total, .1, 3, 70, 200);
%     makes_plot(Im1_1,lineSet_total, 200, on);
%     saveas(gcf,[pwd (strcat(d,num2str(b),e))])
%     makes_plot(Im1_1,lineSet_coord, 200, on);
%     saveas(gcf,[pwd (strcat(d,num2str(b),f))])
    makes_plot(Im1_1,stairLines, 200, on);
    saveas(gcf,[pwd (strcat(d,num2str(b),g))])
end
toc


