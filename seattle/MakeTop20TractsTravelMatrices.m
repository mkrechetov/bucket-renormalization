clc
clear
numRetainedTracts=20;
travelData=table2array(readtable("seattle_travel_numbers.csv"));
sumData=sum(travelData,1);
[~,I]=sort(sumData,'descend');
removingIndices=I(1,numRetainedTracts+1:size(I,2));
removeOrder=sort(removingIndices,'descend');
gisData=readtable("seattle_GIS_data.csv");
gisData_array=table2array(gisData(:,2:5));
uVData=table2array(readtable("seattle_UV_coordinates.csv"));

output_travelData=travelData;
output_gisData=gisData;
output_uVData=uVData;

for i=1:size(removeOrder,2)
    output_travelData(removeOrder(1,i),:)=[];
    output_travelData(:,removeOrder(1,i))=[];
    output_gisData(removeOrder(1,i),:)=[];
    output_uVData(removeOrder(1,i),:)=[];
end
writetable(array2table(output_travelData),strcat('seattle_top',num2str(numRetainedTracts),'_travel_numbers.csv'));
writetable(output_gisData,strcat('seattle_top',num2str(numRetainedTracts),'_GIS_data.csv'));
writetable(array2table(output_uVData),strcat('seattle_top',num2str(numRetainedTracts),'_UV_coordinates.csv'));