function [Et,E0,E1]=ELMTest(W,M,x,y);
    Et=0;E0=0;E1=0;
    [lin,col] = size(y);
    [Ne_Saida,Ne_oculta] = size(y);
    Zk=[];
    bias=-1;
    x=[bias*ones(lin,1) x];
    for Ne=1:lin,
        %loop para os pesos sinapticos ocultos
        Ui=W*x(Ne,:)';
        Zi=1./(1+exp(-Ui));
        Zi=[-1 Zi'];
        Zk=[Zk Zi'];
    end   
    
    
    A=M*Zk;
    for Ne=1:lin,        
        Y=zeros(Ne_Saida,1);
        Y(y(Ne)+1)=1;
        Yt=A(:,Ne);
        
        [out_OK iout_OK]=max(Y);  % Indice da saida desejada de maior valor
        [out_T iout_T]=max(Yt); % Indice do neuronio cuja saida eh a maior
        if iout_OK~=iout_T,   % Conta acerto se os dois indices coincidem 
            if(y(Ne)==0)
                E0=E0+1;
            else
                E1=E1+1;
            end
        end
        Et=E0+E1;
    end       
end