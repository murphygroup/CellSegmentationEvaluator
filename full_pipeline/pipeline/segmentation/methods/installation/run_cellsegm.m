%   =======================================================================================
%   Copyright (C) 2013  Erlend Hodneland
%   Email: erlend.hodneland@biomed.uib.no 
%
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
% 
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
% 
%   You should have received a copy of the GNU General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.
%   =======================================================================================


function[] = run_cellsegm(file_dir, percentage)
%data_dir = '/home/hrchen/Documents/Research/hubmap/data/img'
% img_name = "HBM836.WNJS.587"
% file_name = "random_gaussian_0"
%file_dir = '/data/hubmap/data/MIBI/extracted/Point1';
a = file_dir;
p = str2num(percentage);
minCell = 10*p/100;
maxCell = 50*p/100;
addpath('/home/hrchen/Documents/Research/hubmap/cellsegm');
addpath(a);

% load the data
% load ../data/surfstain_and_nucleus_3D.mat
imnucl = double(read(Tiff('nucleus.tif')));
imsegm = double(read(Tiff('membrane.tif')));

% plane = 5;
% imsegm = imsegm(:,:,plane);
% imnucl = imnucl(:,:,plane);

% Smoothing
prm.smooothim.method = 'dirced';

% No ridge filtering
prm.filterridges = 0;

% threshold for nucleus markers
prm.getminima.nucleus.segmct.thrs.th = 0.50;

% edge enhancing diffusion with a suitable threshold
prm.getminima.nucleus.segmct.smoothim.method = 'eed';
prm.getminima.nucleus.segmct.smoothim.eed.kappa = 0.05;

% method for markers
prm.getminima.method = 'nucleus';

% Subtract the nucleus channel from the surface staining to reduce the
% cross talk effect. 
imsegm1 = imsegm;
filt = fspecial('gaussian',3,2);
imsegm = imfilter(imsegm1,filt) - imfilter(imnucl,filt);

[cellbw,wat,imsegmout,minima,minimacell,info] = ...
    cellsegm.segmsurf(imsegm,minCell,maxCell,'imnucleus',imnucl,'prm',prm);


% 
% cellsegm.show(imsegm,1);title('Surface stain');axis off;
% cellsegm.show(imnucl,2);title('Nucleus stain');axis off;
% cellsegm.show(minima,3);title('Markers');axis off;
% cellsegm.show(wat,4);title('Watershed image');axis off;
% cellsegm.show(cellbw,5);title('Cell segmentation');axis off;
imwrite(uint8(cellbw), strcat(a, filesep, 'mask_cellsegm.png'));
end
