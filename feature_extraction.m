function mapping = feature_extraction(dataset, fisher, retvar, N)

    if fisher == false
       
        if N==0
            [W, N] = pcam(dataset, retvar);
        else
            [W, FRAC] = pcam(dataset, N);
        end
        
        mapping = W;
        
    else
        
        if N==0
            [W1, N] = pcam(dataset, retvar);
            W2 = fisherm(dataset*W1);
        else
            [W1, FRAC] = pcam(dataset, N);
            W2 = fisherm(dataset*W1);
            
        end
            
        
        mapping = W1*W2;

end