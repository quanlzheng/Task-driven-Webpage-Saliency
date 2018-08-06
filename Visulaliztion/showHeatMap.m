load('metric/webpage.mat')

% save file
 method = 'result';
    
if ~exist(['VisualResult2/' method], 'file')
   mkdir(['VisualResult2/' method])
end
for i = 1 : 200
    fprintf('current %d \n', i)
    NumberOfTk = length(webpage{i}.Task);

    for j = 1:NumberOfTk
            
        I = webpage{i,1}.img;
        sal = webpage{i,1}.Task(j).est
            
        ih=heatmap_overlay(I,sal);
        imwrite(ih,fullfile(method, ['image_' num2str(i-1, '%03d') '_task_' num2str(j) '.png']));

    end
        
end





