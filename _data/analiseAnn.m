clear all

folder = "Arritmia/_annotations/";
% folder = "Normal/_annotations/";
files = dir(folder);

names = []; 
annotationData = [];
for f = 1:size(files)
    if files(f).isdir
        continue
   end
   name = string(split(files(f).name, "."));
%    names = [names aux];
   
   file = num2str(folder + name(1));
   format = num2str(name(2));
   annotation = getAnnotationData(file);
    
   annotationData = [annotationData; annotation];
end

save('../_data/analisys.mat', 'annotationData')
% save('../_data/analisys1.mat', 'annotationData')