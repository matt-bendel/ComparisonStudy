% Code to generate acceleration 4 - GRO sampling mask
% shastri.19@buckeyemail.osu.edu

%%
clc; clear all; close all;

%% setting for 360x360 image
dimension = 320;
R = 4;
frames = 1;
mid_sampling_band = 24; % 32 lines in the middle will be fully sampled
buffer = 8;

%%
sampling_lines = (dimension/R);

new_R = dimension/(sampling_lines - mid_sampling_band + buffer);

samp = GoFIX(dimension,frames,new_R);

samp(round(dimension/2) - round(mid_sampling_band/2): round(dimension/2) + round(mid_sampling_band/2)) = 1;

final_acceleration = dimension/nnz(samp)

figure();
imagesc(samp); xlabel('frames'); ylabel('PE'); %axis('image'); %colormap(c); %colorbar;

[index, val] = find(samp~=0);

csvwrite('gro_r4_mask.csv',samp)

