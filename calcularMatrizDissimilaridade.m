function D1 = calcularMatrizDissimilaridade(dataframe, idx_colunas_numericas, idx_colunas_ordinais, idx_colunas_categoricas, idx_distribuicoes_binarias_simetricas)

%     idx_colunas_numericas = [1 2 5 6]; idx_colunas_ordinais = [3]; idx_colunas_categoricas = [4 7:27]; idx_distribuicoes_binarias_simetricas = [11 20];

    D = zeros(size(dataframe,1));
    D1 = zeros(size(dataframe,1));
    
    D2 = zeros(size(dataframe));
    
    numero_categorias = 16;

    for idx_reg = 1:size(D,1)
        for idx_col = 1:size(dataframe,2);
            aux = repmat(dataframe(idx_reg,idx_col),size(dataframe,1),1);
            pesos_delta = ones(size(dataframe));
            if ismember(idx_col,idx_colunas_numericas)
                d = (aux - dataframe(:,idx_col))/(max(dataframe(:,idx_col)) - min(dataframe(:,idx_col)));
            else if ismember(idx_col,idx_colunas_ordinais)
                    z1 = (aux - 1)/(numero_categorias - 1);
                    z2 = (dataframe(:,idx_col) - 1)/(numero_categorias - 1);
                    d = abs(z1-z2);
                else if ismember(idx_col,idx_colunas_categoricas)
                        d = double(~(aux == dataframe(:,idx_col)));
                        if ~ismember(idx_col,idx_distribuicoes_binarias_simetricas)
                            pesos_delta((aux==0)&(dataframe(:,idx_col)==0), idx_col) = 0;
                        end
                    end
                end
            end
            D2(:,idx_col) = d;
        end
        D1(:,idx_reg) = sum((pesos_delta .* D2), 2)./sum(pesos_delta, 2);
    end
    
%     for idx_reg1 = 1:size(D,1)
%         for idx_reg2 = (idx_reg1+1):size(D,1)
%             pesos_delta = ones(1,size(D,2));
%             d = pesos_delta;
%             for idx_col = 1:size(dataframe,2)
%                 if ismember(idx_col, idx_colunas_numericas)
%                     d(idx_col) = abs(dataframe(idx_reg1,idx_col) - dataframe(idx_reg2,idx_col))/(max(dataframe(:,idx_col)) - min(dataframe(:,idx_col)));
%                 else if ismember(idx_col,idx_colunas_ordinais)
%                         z1 = (dataframe(idx_reg1,idx_col) - 1)/(numero_categorias - 1);
%                         z2 = (dataframe(idx_reg2,idx_col) - 1)/(numero_categorias - 1);
%                         d(idx_col) = abs(z1-z2);
%                     else if ismember(idx_col,idx_colunas_categoricas)
%                         if dataframe(idx_reg1,idx_col) == dataframe(idx_reg2,idx_col)
%                             d(idx_col) = 0;
%                         else
%                             d(idx_col) = 1;
%                         end
%                         
%                         if (~ismember(idx_col,idx_distribuicoes_binarias_simetricas)) && (dataframe(idx_reg1,idx_col) == 0) && (dataframe(idx_reg2,idx_col))
%                             pesos_delta(idx_col) = 0;
%                         end
%                         end
%                     end
%                 end
%             end
%         end
%     end
end