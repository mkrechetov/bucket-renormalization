clc
clear
rawData=readmatrix("RawData.csv");
countyTravel(size(rawData,1)-1,size(rawData,1)-1)=0;
for i=1:size(countyTravel,1)
    for j=1:size(countyTravel,1)
        if i~=j
            countyTravel(i,j)=rawData(i,3)*rawData(j,6);
        end
    end
end
for i=size(countyTravel,1):-1:1
    for j=size(countyTravel,1):-1:1
        avg=countyTravel(i,j)+countyTravel(j,i)/2;
        countyTravel(j,i)=round(avg);
        countyTravel(i,j)=round(avg);
    end
end
sumTravel=sum(sum(countyTravel));
countyTravelProbabilities=countyTravel./sumTravel;
writematrix(countyTravelProbabilities,"wisconsin_travel_probabilities.csv");
countyTravelDaily=countyTravel./365;
writematrix(countyTravelDaily,"wisconsin_travel_numbers.csv");