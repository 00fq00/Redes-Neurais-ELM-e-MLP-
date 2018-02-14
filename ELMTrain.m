function [W,M]=ELMTrain(Ne_oculta,Ne_Saida,epocas,x,y);
    bias=-1;
    [lin,col] = size(x);
    x=[bias*ones(lin,1) x];
    a=0; b=0.1; % define intervalo dos pesos
    [lin,col] = size(x);
    %W=a+(b-a).*rand(Ne_oculta,col);
    sig=0.1; % define desvio-padrao dos pesos
    W=sig*randn(Ne_oculta,col); % gera numeros gaussianos
    M=[];
    D=zeros(lin,Ne_Saida);
    
    for t=1:epocas,
            %para cada individuo no banco de dados
            D=zeros(lin,Ne_Saida);
            I=randperm(lin);
            x=x(I,:);
            y=y(I,:);
            Zk=zeros(Ne_oculta+1,lin);

            for Ne=1:lin,
                %Foward Propagation
                D(Ne,y(Ne)+1)=1;
                %loop para os pesos sinapticos ocultos
                Ui=W*x(Ne,:)';
                Zi=1./(1+exp(-Ui));
                Zi=[-1 Zi'];
                Zk(:,Ne)=Zi;
            end    
            D=D';
            M = D*pinv(Zk);
    end
end