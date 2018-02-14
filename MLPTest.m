function [Et,E0,E1]=MLPTest(W,M,x,y);
    bias=-1;
    Et=0;E0=0;E1=0;
    [lin,col] = size(y);
    [Ne_oculta,col] = size(W);
    z1=zeros(1,Ne_oculta);
    nClass=max(y)+1;
    SaidaCorreta=zeros(lin,nClass);
    Ne_Saida=nClass;
    %coloando o bias no vetor de entrada x
    x=[bias*ones(lin,1) x];
    for Ne=1:lin,
        %Foward Propagation
        %loop para os pesos sinapticos ocultos
        Ui=W*x(Ne,:)';
        Zi=1./(1+exp(-Ui));
        Et=0;E0=0;E1=0;
        Zk=[-1 Zi'];
        %camada de saida
        Uk=M*Zk';
        Yt=1./(1+exp(-Uk));

        %Back propagation
        %erro da saida esperada
        Y=zeros(Ne_Saida,1);
        Y(y(Ne)+1)=1;
        
        
        
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