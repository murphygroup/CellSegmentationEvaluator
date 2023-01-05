%%%%--------ONLY WHEN USED FOR FIRST TIME --------------
% 1. go to folder ../core/mscripts
% 2. type to the command line: makeMex
%%%%----------------------------------------------------

function[] = run_CellX(file_name)

try
% clear; clc; close all;

% paths for core-code
thisFile = mfilename('fullpath');
[folder, name] = fileparts(thisFile);
cd(folder);
addpath('core/mscripts');
addpath('core/mclasses');
addpath('core/mfunctions');
addpath('core/mex');
addpath('core/mex/maxflow');
% path for code of other algos
addpath(genpath('examples_singleImages/extraFunctions'));

% img_name = 'HBM836.WNJS.587';
% file_name = 'random_gaussian_0';
% DEFINE THE IMAGE-FILES TO BE SEGMENTED

% fileTags = {'R001_X001_Y001_c7z5_CD16_membrane_original.ome'}; %{'R001_X001_Y001_c40z5_HLA-ABC_cytoplasm_original.ome'} 

    
% set paths
% fileTag=fileTags{imn};
rawImgFolder = [file_name];
resultFolder = rawImgFolder;

% if ~exist(resultFolder,'dir')
%     mkdir(resultFolder)
% end
fprintf('running file %s \n', file_name)
    
% load files
% inputs: the images
imgFileName = ['membrane.tif'];
imgSegFileNameN = [rawImgFolder filesep imgFileName];
segImage = imgSegFileNameN;
    

% set calibration file
calibrationfilename = append(file_name, '/CellX_config.xml')

config = CellXConfiguration.readXML(calibrationfilename);
%     if strcmp(fileTag,'phase_03')
%         %config.setIdPrecisionRate(0.9);
%     elseif strcmp(fileTag,'BF_position040521_time0001')
%         %config.setIdPrecisionRate(0);
%     elseif strcmp(fileTag,'TransNS1_050004')
%         %config.setIdPrecisionRate(1);
%     elseif strcmp(fileTag,'ph2_position010100_time0001')
%         %config.setIdPrecisionRate(0.1);
%     elseif strcmp(fileTag,'BF_position040511_time0001')
%         %config.setIdPrecisionRate(0);
%         
%     end
    
config.setDebugLevel(1);

config.check();
    
% get file set
frameNumber = 1;    
fileSet = CellXFileSet(frameNumber, segImage);
fileSet.setResultsDirectory(resultFolder);
    
% Run segmentation
seg = CellXSegmenter(config, fileSet);
seg.run();
segmentedCells =seg.getDetectedCells();
    
    
%------SAVE RESULTS----------
% write final images
writeSegmImages(config, fileSet, seg, segmentedCells)
%     
% % write initial image
% finame = [resultFolder filesep 'initImage.tif'];
% imwrite(seg.image,finame,'tif')
% fprintf('Wrote %s\n', finame);
%     
% % save segmented cells
% savefileneCellX = fileSet.seedsMatFile;
% save(savefileneCellX,'segmentedCells')
% fprintf('Wrote %s\n', savefileneCellX);
%     
%    
% % produce segmentation mask
% segmMask=zeros(size(seg.image));
% for nsc=1:numel(segmentedCells)
%     cellPixelInd = segmentedCells(nsc).cellPixelListLindx;
%     segmMask(cellPixelInd)=nsc;   
% end
% 
% 
% % and save it
% savefileneCellXmask = fileSet.maskMatFile;
% save(savefileneCellXmask,'segmMask')
% fprintf('Wrote %s\n', savefileneCellXmask);
    
% % write TXT result of the current segmentation
% CellXResultWriter.writeTxtSegmentationResults(...
% fileSet, ...
% segmentedCells, ...
% config);
% fprintf('Wrote %s\n', fileSet.seedsTxtFile);
% 
% CellXResultWriter.writeSeedingControlImage( ...
% segImage, ...
% fileSet.seedingImageFile, ...
% seg.seeds, ...
% config...
% );
% CellXResultWriter.writeSegmentationControlImageWithIndices( ...
% segImage, ...
% fileSet.controlImageFile, ...
% segmentedCells, ...
% config...
% );
% end
catch
	exit()
end
end
