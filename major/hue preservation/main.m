
clc;
close all;
clear all;
warning off
%%%%%under water image enhncement

    %%%%%%%%%%%%%%%% Image Aquisition %%%%%%%%%%%%%%%%%%%%%%%%%%%
[filename, pathname] = uigetfile({'*.jpg';'*.png'}, 'pick an image');
if isequal(filename, 0) || isequal(pathname, 0)   
    helpdlg('Image input canceled.');  

else
    X=imread(fullfile(pathname, filename));
end

figure,imshow(X);title('original  image');
%%%%%%%%%%%%devide color space images
  N = 256;
    A = im2uint8(X);
    a=rgb2hsv(X);
    fig = figure;
    
    subplot(2,2,1);
    imshow(A);
    title('original image');
    ColorList = { 'Red' 'Green' 'Blue' };
    gr = 0:1/(N-1):1;        
    
    for k = 1:3
        % color map:
        cMap = zeros(N,3);
        cMap(:,k) = gr;
        subplot(2,2,k+1);
        imshow(ind2rgb(A(:,:,k),cMap));
        title(ColorList{k});
        
    end
%%% bidirectional Empirical Mode Decomposition (BEMD) analysis
normal_thr_limit=0.5;
low_limt=0.002;
up_limit=0.999;
%----------------------------------------------------------------------
under_image=X;
[CONTRAST saliency chromatic]=size(X);
%----------------------------------------------------------------------
% apply CNN 
[nndata AlexNet]=size(chromatic);
 nn_data_image=0:0.1:nndata;
y=nn_data_image.^3;
net=newff(minmax(nn_data_image),[20,AlexNet],{'logsig','purelin','trainln'});
net.trainparam.epoches=4000;
net.trainparam.goal=1e-25;
net.trainparam.lr=0.01;
net=train(net,nn_data_image,y);

out_nn=y(11);
out_nn=net(nn_data_image(11));
nn_data=(ceil(out_nn)./out_nn);
out_AlexNet=floor(nn_data);
G=[];

if chromatic==3
    inc_pixel_limit=0.04;dec_pixel_limt=-0.04;
    max_chromatic=rgb2ntsc(under_image);
    mean_adjustment=inc_pixel_limit-mean(mean(max_chromatic(:,:,2)));
    max_chromatic(:,:,2)=max_chromatic(:,:,2)+mean_adjustment*(out_AlexNet-max_chromatic(:,:,2));
    mean_adjustment=dec_pixel_limt-mean(mean(max_chromatic(:,:,3)));
    max_chromatic(:,:,3)=max_chromatic(:,:,3)+mean_adjustment*(out_AlexNet-max_chromatic(:,:,3));
else
    max_chromatic=double(under_image)./255;
end
%----------------------------------------------------------------------
mean_adjustment=normal_thr_limit-mean(mean(max_chromatic(:,:,1)));
max_chromatic(:,:,1)=max_chromatic(:,:,1)+mean_adjustment*(out_AlexNet-max_chromatic(:,:,1));
if chromatic==3
    max_chromatic=ntsc2rgb(max_chromatic);
end
%----------------------------------------------------------------------
under_image=max_chromatic.*255;
%--------------------caliculate the min to max pixels----------------------
for k=1:chromatic
    arr=sort(reshape(under_image(:,:,k),CONTRAST*saliency,1));
    saliency_min(k)=arr(ceil(low_limt*CONTRAST*saliency));
    luminance_max(k)=arr(ceil(up_limit*CONTRAST*saliency));
end
%----------------------------------------------------------------------
if chromatic==3
    saliency_min=rgb2ntsc(saliency_min);
    luminance_max=rgb2ntsc(luminance_max);
end
%----------------------------------------------------------------------
under_image=(under_image-saliency_min(1))/(luminance_max(1)-saliency_min(1));
figure,imshow(under_image);title(' image enhancement');

    
% % %%%%%%%%%%%%%%%temporal correlation pixels separation
red_color_correlation = adapthisteq(under_image(:,:,1));
            green_adapthisteq_green = adapthisteq(under_image(:,:,2));
            blue_adapthisteq_blue = adapthisteq(under_image(:,:,3));
            fusion_enhanced = cat(3,red_color_correlation,green_adapthisteq_green,blue_adapthisteq_blue);
            figure,imshow(mat2gray(fusion_enhanced));title(' output image');
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%% singlr part extrraction
            
%             single_partr=imcrop(fusion_enhanced);
%                         figure,imshow(mat2gray(single_partr));title('particular part of image');

            