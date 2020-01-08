%% Justin Ngo
% Computer Vision Assignment 2: Camera Calibration and Pose Estimation

clear, clc, close all

%% Problem 1: Calibration 
% calib_gui

%% Problem 2: simple pose estimation model
% step 1: using undistor_sequence in calib_gui I obtain an distortion
% corrected image. i.e. left_rect1

% import undistorted image (1920 - width, 1080 height)
% real world squares are 30mm each
I = imread('left_rect2.bmp');

% camera clibration matrix K (all in pixels ~ obtained in 1)
K = [1443 0 902; 0 1446 470; 0 0 1];

% select a set of 10 points that you know 3d location i.e 4 corners 
% of the inner square in the picture can be given as (0,0)(5,0)(5,7)(0,7) 
% forgoing the outer squares, try to click as close as possible. It just to
% scale
% NOTE: origin(0,0) is in top left
% ----------> 1920
% |
% |
% | 
% v
% 1080
imshow(I)
q = ginput(10); %2D coordinates
homo_q = [1 1 1 1 1 1 1 1 1 1]';
q = [q homo_q];
close 
Qm = [0 5 5 0 1 4 4 1 2 3 ; 0 0 7 7 1 1 6 6 2 3 ;...
     0 0 0 0 0 0 0 0 0 0 ]'; % world coordinates (x,y,z(0 cause obj flat))
% hold on
% plot(q(:,1),q(:,2),'LineWidth',3)

% select 3 2D coordinate, q, that does not lie on a straight line. first 3
% is a good canidate. Form x,y,1 - homogenous
q1 = q(1,:)';
q2 = q(2,:)';
q3 = q(3,:)';

% find pairwise distances from the points in the world coordinates
d12 = norm(Qm(1,:) - Qm(2,:));
d23 = norm(Qm(2,:) - Qm(3,:));
d31 = norm(Qm(1,:) - Qm(3,:));

% from Qc (camera coordinate) = lambda_i*K^(-1)*qi
% and using pairwise distances we calculate lambda 1, 2, 3
% K^(-1)*q_i = (Xi,Yi,Zi), where i = 1,2,3

syms lambda1 lambda2 lambda3
r = K\q1;
s = K\q2;
t = K\q3;

a = (lambda1*r(1) - lambda2*s(1))^2 + (lambda1*r(2) - lambda2*s(2))^2 + ...
    (lambda1*r(3) - lambda2*s(3))^2 == d12^2;
b = (lambda2*s(1) - lambda3*t(1))^2 + (lambda2*s(2) - lambda3*t(2))^2 + ...
    (lambda2*s(3) - lambda3*t(3))^2 == d23^2;
c = (lambda3*t(1) - lambda1*r(1))^2 + (lambda3*t(2) - lambda1*r(2))^2 + ...
    (lambda3*t(3) - lambda1*r(3))^2 == d31^2;

solution = solve(a, b, c);

lambda1_ = double(solution.lambda1)';
lambda2_ = double(solution.lambda2)';
lambda3_ = double(solution.lambda3)';

% keep real solutions and non-negative
j = 1;
for i = 1:length(lambda1_)
    if abs(imag(lambda1_(i))) > 1.0e-100
        continue
    end
    if real((lambda1_(i)) >= 0)
        L1(j) = real(lambda1_(i));
        j = j + 1;
    end
end
j = 1;
for i = 1:length(lambda2_)
     if abs(imag(lambda2_(i))) > 1.0e-100
        continue
    end
    if real((lambda2_(i)) >= 0)
        L2(j) = real(lambda2_(i));
        j = j + 1;
    end
end
j = 1;
for i = 1:length(lambda3_)
     if abs(imag(lambda3_(i))) > 1.0e-100
        continue
    end
    if real((lambda3_(i)) >= 0)
        L3(j) = real(lambda3_(i));
        j = j + 1;
    end
end

% comopute camera coordinate of the object Qc, where Qc = lambda*[Xi,Yi,Zi]
% where [Xi,Yi,Zi] = inverse(K)*qi

for i = 1:length(L1)
   Qc1(:,i) = L1(i)*r; 
end
for i = 1:length(L2)
   Qc2(:,i) = L2(i)*s; 
end
for i = 1:length(L3)
   Qc3(:,i) = L3(i)*t; 
end
% compute the rotation and translation
i = 1;
mean_RPE = 0; % mean reporjection error
tot_mean_RPE = 0;
low_RPE = 10; % smallest reprojection error

% save the lambdas that is lowest
saveL1 = 1;
saveL2 = 1;
saveL3 = 1;
savePose = 1;
for u = 1:length(L1)
    for v = 1:length(L2)
        for w = 1:length(L3)
            FinalTrans{i} = Register3DPointsQuaternion([Qm(1,:);Qm(2,:);...
                Qm(3,:)]',[Qc1(:,u) Qc2(:,v) Qc3(:,w)]);
            R{i} = FinalTrans{i}(1:3,1:3);
            T{i} = (-1*R{i})'*FinalTrans{i}(1:3,4);
            % compute mean reporjection using R - Rotation, T - Translation
            % for all 10 2D pixels q by projecting world points Qm. Find
            % lambdas that produces smallest mean projection error
            for n = 1:length(Qm)
                 q_re(:,n) = K*(R{i}*Qm(n,:)' - R{i}*T{i}); %reprojection q
                 % remove scaling factor. So you obtain (u;v;w) you then
                 % want (u/w;v/w;1)
                 q_re(:,n) = q_re(:,n)/q_re(3,n); 
                 mean_RPE = sqrt((q(n,1) - q_re(1,n))^2 + ...
                                 (q(n,2) - q_re(2,n))^2);
                 tot_mean_RPE = tot_mean_RPE + mean_RPE;
            end
            tot_mean_RPE = tot_mean_RPE/10;
            if tot_mean_RPE < low_RPE
                low_RPE = mean_RPE;
                saveL1 = u;
                saveL2 = v;
                saveL3 = w;
                savePose = i;
            end
            
            i = i + 1;
            
        end
    end
end

%plot3(Qm(:,1)*102.46,Qm(:,2)*102.46,Qm(:,3),'k*'); %102.46 PPI wiki 1080p
%hold on
%plot3(q(:,1),q(:,2),q(:,3),'r*');
plot(q(:,1),q(:,2),'r*');
hold on
% plot3(q_re(1,:),q_re(2,:),q_re(3,:),'g*');
plot(q_re(1,:),q_re(2,:),'g*');
title('3D plot of reprojection')
ylabel('1080 pixels')
zlabel('height')
xlabel('1920 pixels')
% print pose and other things
fprintf('Reprojection Error = %0.3f\n', low_RPE)
disp('Rotation = ')
disp(R{savePose})
disp('Translation = ')
disp(T{savePose})
fprintf('Lambda 1 = %0.3f\n', L1(saveL1))
fprintf('Lambda 2 = %0.3f\n', L2(saveL2))
fprintf('Lambda 3 = %0.3f\n', L3(saveL3))

%% Problem 3 Image based localization
% commands
% sift -> extract SIFT features and return matrixform
% [image, descrips, locs] = sift('scene.pgm')
% showkeys -> to desiplay keypoints 
% showkeys(image, locs);
% match('scene.pgm','book.pgm')
% VOCABULARY TREE PARAMETERS: Branch factor 4, with 3 LEVELS

% must have sift32.exe in the same folder as the code

%% Bag of Words
clear, clc, close all
% extract features
k = 1000;
query = dir(['query' '/*.png']);
qlen = size(query,1);
database = dir(['database' '/*.png']);
dlen = size(database,1);
success = 0;
for i = 1:qlen
    successflag = 0;
% I = imread(filename);
% imshow(I)
% [ image , descriptors, locs ] = sift(filename);
    [ ~ , q_descriptors, ~] = siftLowe(query(i).name);
% [idx, C] = kmeans(q_descriptors, k);
    % C (centers) is 128 by k, these are now the vocabulary words
    % A (assignments) is number of assignments of q_descriptors to k (number of centers)
    % vl_kmeans is used instead of in house kmeans cause its much faster
    if length(q_descriptors) < k
        [C,A] = vl_kmeans(q_descriptors',length(q_descriptors));
    else
        [C,A] = vl_kmeans(q_descriptors',k);
    end
    % build histogram of the features and note them
    picname = [];
    % compare each database image to query to find the match with the
    % features found above. 
    for j = 1:dlen 
        num_match = 0;
        [ ~ , d_descriptors, ~] = siftLowe(database(j).name);
        % distRatio: Only keep matches in which the ratio of vector angles from the
        %   nearest to second nearest neighbor is less than distRatio.
        distRatio = 0.6;   
        % For each descriptor in the first image, select its match to second image.
        des2t = d_descriptors';                          % Precompute matrix transpose
        for w = 1 : size(q_descriptors,1)
           dotprods = q_descriptors(w,:) * des2t;        % Computes vector of dot products
           [vals,indx] = sort(acos(dotprods));  % Take inverse cosine and sort results

           % Check if nearest neighbor has angle less than distRatio times 2nd.
           if (vals(1) < distRatio * vals(2))
              match(w) = indx(1);
           else
              match(w) = 0;
           end
        end
        for x = 1:length(match)
           if match(x) ~= 0;
               num_match = num_match + 1;
           end
        end
        picname(j) = num_match;
    end
    [top5, top5_index] = sort(picname,'descend');
    top5_index = top5_index(1,1:5);
    % determine the sucess of this query. 
    for y = 1:5
        retrieved = database(top5_index(y)).name;
        %get rid of the numbers following after
        for z = 1:4
            under_num = sprintf('_%d',z);
            retrieved = strrep(retrieved,under_num,'');
        end
        if strcmp(query(i).name,retrieved)
            successflag = successflag + 1;
        end
    end
    if successflag > 0
        success = success + 1;
    end
end

accuracyBOW = 100*success/qlen;
fprintf('\n Accuracy of Bag of Words is %.2f percent\n', accuracyBOW);

%% Vocabulary Tree
% branch factor 4 "k"
% levels 3
clear, clc, close all
query = dir(['query' '/*.png']);
qlen = size(query,1);
database = dir(['database' '/*.png']);
dlen = size(database,1);
% k means tree with sift descriptors
% build vocabulary tree from database 
database_desc = [];
k = 4; % branching factor
depth = 3;
min_vocab_size = k^depth;

if ~exist('vocabulary_tree.mat')
	for i = 1:dlen
        img = single(rgb2gray(imread(database(i).name)));
        [sift_frames, sift_desc] = vl_sift(img);
        database_desc = [database_desc sift_desc];
    end
    vocabulary_tree = vl_hikmeans(database_desc, k, min_vocab_size);
	%save vocab tree as a file....
	save('vocabulary_tree.mat','vocabulary_tree');
else 
	load('vocabulary_tree.mat', 'vocabulary_tree');    
end


% map image to vocab tree using L1 or L2 (euclidean) to cluster center by
% sift

% Computing the Virtual Inverted File Index
% 
% Precomputing node weights (entropy)
% Precomputing norms of d-vectors representing DB images

if ~exist('invfindex.mat')
	[invfindex, img_norms, node_weights, dbImgNames]=inverted_file_index('database', vocabulary_tree);
	%save db_vectors as a file....
	save('invfindex.mat','invfindex');
	save('img_norms.mat','img_norms');
	save('node_weights.mat','node_weights');
	%save the order of imgs and their ids (indices in dbImgNames) just in case
	%order read changes, or an image gets lost
	save('dbImgNames.mat','dbImgNames'); 
else
	load('invfindex.mat','invfindex');
	load('img_norms.mat','img_norms');
	load('node_weights.mat','node_weights');
	load('dbImgNames.mat','dbImgNames');
end

success = 0;
for s = 1:qlen
    successflag = 0;
    queryimg = single(rgb2gray(imread(query(s).name)));
    [candidates,scores]=run_query(queryimg, vocabulary_tree, invfindex, img_norms, node_weights);
    % determine the sucess of this query. 
    fprintf('~~~~ %d ~~~~~\n', s);
    fprintf('%s query\n',query(s).name);
    for y = 1:5
        retrieved = database(candidates(y)).name;
        fprintf('%s\n',retrieved);
        %get rid of the numbers following after
        for z = 1:4
            under_num = sprintf('_%d',z);
            retrieved = strrep(retrieved,under_num,'');
        end
        if strcmp(query(s).name,retrieved)
            successflag = successflag + 1;
        end
    end
    if successflag > 0
        success = success + 1;
        fprintf('success\n');
    end
     fprintf('~~~~~~~~~~~\n');
end
accuracyVocabTree = 100*success/qlen;
fprintf('\n Accuracy of vocabulary tree is %.2f percent\n', accuracyVocabTree);
