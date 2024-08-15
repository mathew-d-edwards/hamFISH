function [] = correctLumi(correct_image_fp, processPath, outputPath)

    % this function will generate the corrected lumi dapi and put them into
    % viewedtiles under stella
    
    %load dapi files
    originalDapiFiles = dir(fullfile(processPath, '*/hyb1_dapi.tif'));
    
    %output path
    if ~exist(outputPath, 'dir')
        mkdir(outputPath)
    end
    
    % default path to the correct mask
    scale_mask = Tiff(correct_image_fp);
    scale_mask = read(scale_mask);
    scale_mask = double(scale_mask);
    background_scale = double(scale_mask ./ max(max(scale_mask)));
   

    % iterate through each dapi and process them
    for k = 1:length(originalDapiFiles)

        %extract zscanid so we know what we are processing
        zscan_id = strsplit(originalDapiFiles(k).folder, '_');
        zscan_id = strcat('zscan_', zscan_id{end});
        
        % set output filename
        outputfilename = strcat(zscan_id, '_hyb1_dapi_corrected.tif');
        completeFilePath = fullfile(originalDapiFiles(k).folder, originalDapiFiles(k).name);

        % read in dapi tif
        current_tiff = Tiff(completeFilePath);
        current_tiff = read(current_tiff);
        current_tiff = double(current_tiff);
%         imshow(current_tiff, [0 5000])
%         
%         %
%         se = strel('disk', 200);
%         background = imopen(current_tiff, se);
%         imshow(background, [0 5000]);
%         
%         %
%         test_tiff1 = current_tiff - (background * 1);
%         imshow(test_tiff1, [0 5000]);
%         
        % log it 
        tif1 = log2(current_tiff);
%         imshow(tif1, [0 15]);
        
        % clear background first
        se = strel('disk', 50);
        background = imopen(tif1, se);
        %imshow(background, [0 10]);
        
        % subtract background
        test_tiff1 = tif1 - (background * 1);
%         imshow(test_tiff1, [0 5]);
        
        % correct shading 
        test_tiff2 = test_tiff1 ./ background_scale;
%         imshow(test_tiff2, [0 5]);

        % multiply 
        test_tiff3 = 100 * test_tiff2;
%         imshow(test_tiff3, [0 500]);
        
        % exp back
        test_tiff4 = test_tiff3 .^ 1.1;
%         imshow(test_tiff4, [0 1000]);
%         
%         % exp back
%         test_tiff4 = 1.7 .^ test_tiff3;
%         imshow(test_tiff4, [0 8]);
% 
%         
%         test_tiff5 = 100 * test_tiff4;
%         imshow(test_tiff5, [0 500]);
        
        %save
        use_tiff = test_tiff4;
        
        final_tiff = uint8(255 * mat2gray(use_tiff, [0, double(mean(quantile(use_tiff, 0.998)))* 1.5]));

        imshow(final_tiff)
        
        imwrite(final_tiff, fullfile(outputPath, outputfilename));
    end
end