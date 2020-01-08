function [ ncc_score ] = NCC( patch1, patch2 )
% computes normalized cross correlation for 3d computer vision
% by Lukas Gabert

[ywidth, xwidth] = size(patch1);

num = 0;
asum = 0;
bsum = 0;
for i = 1:ywidth
    for j = 1:xwidth
        num = num + double(patch1(i,j))*double(patch2(i,j));
        asum = asum + double(patch1(i,j))^2;
        bsum = bsum + double(patch2(i,j))^2;
    end
end

% num = double(num);
% asum = double(asum);
% bsum = double(bsum);
ncc_score = num/(sqrt(asum)*sqrt(bsum));

end

