for i=10
    csvwrite(['chb06_' num2str(i) '.csv'],rdsamp(['chb06_' num2str(i) '.edf'],[],3686400,1,0,0)) 
end
for i=24
    csvwrite(['chb06_' num2str(i) '.csv'],rdsamp(['chb06_' num2str(i) '.edf'],[],3686400,1,0,0)) 
end
