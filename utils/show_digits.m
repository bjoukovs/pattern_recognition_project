function show_digits(digits, labels)
    
    %Plot custom classified digits
    
    figure;
    
    n = length(labels)
    
    for i=1:n
       subplot(ceil(n/5),5,i);
       
       digit = digits(i,:);
       img = reshape(digit(:), [64,64]);
       imshow(img);
       
       xlabel(labels(i,end));
       
    end
    
    
    
end

