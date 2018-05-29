
# coding: utf-8

# # Redes Neurais

# In[1]:


# Processamento dos dados
import numpy  as np
import pandas as pd


# In[2]:


# Visualização dos dados
import matplotlib.pyplot as plt
import seaborn           as sns

# In[3]:

# Seleção dos modelos
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV

# In[4]:

# Armazenamento externo de arquivos
from sklearn.externals import joblib


# In[5]:


# Métricas de avaliação
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report


# ## Carregamento dos datasets balanceados

# In[6]:


# Conjuntos de treinamento
# Conjunto 1
conjunto1_train        = pd.read_csv('Data/datasets-balanceados/train_data_b_1.csv')
conjunto1_train_target = pd.read_csv('Data/datasets-balanceados/train_target_b_1.csv')

# Conjunto 2
conjunto2_train        = pd.read_csv('Data/datasets-balanceados/train_data_b_2.csv')
conjunto2_train_target = pd.read_csv('Data/datasets-balanceados/train_target_b_2.csv')

# Conjunto 3
conjunto3_train        = pd.read_csv('Data/datasets-balanceados/train_data_b_3.csv')
conjunto3_train_target = pd.read_csv('Data/datasets-balanceados/train_target_b_3.csv')

# Conjunto 4
conjunto4_train        = pd.read_csv('Data/datasets-balanceados/train_data_b_4.csv')
conjunto4_train_target = pd.read_csv('Data/datasets-balanceados/train_target_b_4.csv')


# In[7]:


# Conjuntos de teste
# Conjunto 1
conjunto1_test        = pd.read_csv('Data/datasets-balanceados/test_data_b_1.csv')
conjunto1_test_target = pd.read_csv('Data/datasets-balanceados/test_target_b_1.csv')

# Conjunto 2
conjunto2_test        = pd.read_csv('Data/datasets-balanceados/test_data_b_2.csv')
conjunto2_test_target = pd.read_csv('Data/datasets-balanceados/test_target_b_2.csv')

# Conjunto 3
conjunto3_test        = pd.read_csv('Data/datasets-balanceados/test_data_b_3.csv')
conjunto3_test_target = pd.read_csv('Data/datasets-balanceados/test_target_b_3.csv')

# Conjunto 4
conjunto4_test        = pd.read_csv('Data/datasets-balanceados/test_data_b_4.csv')
conjunto4_test_target = pd.read_csv('Data/datasets-balanceados/test_target_b_4.csv')


# ## Carregamento dos datasets desbalanceados

# In[8]:


# Leitura dos Datasets desbalanceados

# Conjuntos de treinamento

# Conjunto 1
conjunto1_train_desbalanceado        = pd.read_csv('Data/datasets-desbalanceados/train_data_1.csv')
conjunto1_train_target_desbalanceado = pd.read_csv('Data/datasets-desbalanceados/train_target_1.csv')

# Conjunto 2
conjunto2_train_desbalanceado        = pd.read_csv('Data/datasets-desbalanceados/train_data_2.csv')
conjunto2_train_target_desbalanceado = pd.read_csv('Data/datasets-desbalanceados/train_target_2.csv')

# Conjunto 3
conjunto3_train_desbalanceado        = pd.read_csv('Data/datasets-desbalanceados/train_data_3.csv')
conjunto3_train_target_desbalanceado = pd.read_csv('Data/datasets-desbalanceados/train_target_3.csv')

# Conjunto 4
conjunto4_train_desbalanceado        = pd.read_csv('Data/datasets-desbalanceados/train_data_4.csv')
conjunto4_train_target_desbalanceado = pd.read_csv('Data/datasets-desbalanceados/train_target_4.csv')


# In[9]:


# Conjuntos de teste
# Conjunto 1
conjunto1_test_desbalanceado        = pd.read_csv('Data/datasets-desbalanceados/test_data_1.csv')
conjunto1_test_target_desbalanceado = pd.read_csv('Data/datasets-desbalanceados/test_target_1.csv')

# Conjunto 2
conjunto2_test_desbalanceado        = pd.read_csv('Data/datasets-desbalanceados/test_data_2.csv')
conjunto2_test_target_desbalanceado = pd.read_csv('Data/datasets-desbalanceados/test_target_2.csv')

# Conjunto 3
conjunto3_test_desbalanceado        = pd.read_csv('Data/datasets-desbalanceados/test_data_3.csv')
conjunto3_test_target_desbalanceado = pd.read_csv('Data/datasets-desbalanceados/test_target_3.csv')

# Conjunto 4
conjunto4_test_desbalanceado        = pd.read_csv('Data/datasets-desbalanceados/test_data_4.csv')
conjunto4_test_target_desbalanceado = pd.read_csv('Data/datasets-desbalanceados/test_target_4.csv')


# ## Pré-processamento dos dados

# ### Remoção de colunas inúteis

# In[10]:


# Lista de csvs
df_data_balanceado        = [conjunto1_train, conjunto2_train ,conjunto3_train ,conjunto4_train,
                               conjunto1_test, conjunto2_test  ,conjunto3_test  ,conjunto4_test]

df_target_balanceado      = [conjunto1_train_target, conjunto2_train_target, conjunto3_train_target, conjunto4_train_target,
                              conjunto1_test_target, conjunto2_test_target , conjunto3_test_target , conjunto4_test_target]

df_data_desbalanceado     = [conjunto1_train_desbalanceado, conjunto2_train_desbalanceado, conjunto3_train_desbalanceado, conjunto4_train_desbalanceado,
                              conjunto1_test_desbalanceado,  conjunto2_test_desbalanceado,  conjunto3_test_desbalanceado, conjunto4_test_desbalanceado]

df_target_desbalanceado   = [conjunto1_train_target_desbalanceado, conjunto2_train_target_desbalanceado, conjunto3_train_target_desbalanceado, conjunto4_train_target_desbalanceado,
                              conjunto1_test_target_desbalanceado,  conjunto2_test_target_desbalanceado,  conjunto3_test_target_desbalanceado, conjunto4_test_target_desbalanceado]


# In[11]:


# Remoção das colunas ids
for df_index in range(0,8):
    # Dataset balanceado
    df_data_balanceado[df_index].drop('id',axis=1,inplace=True)
    df_target_balanceado[df_index].drop('id',axis=1,inplace=True)
    
    # Dataset desbalanceado
    df_data_desbalanceado[df_index].drop('id',axis=1,inplace=True)
    df_target_desbalanceado[df_index].drop('id',axis=1,inplace=True)


# In[12]:


# Transformação das features categóricas em dummy variables, utilizando get_dummies

def get_dummies_features(df):
    ''' Função utilizada para a codificação de features categóricas de um dataframe em features binárias'''
    output = pd.DataFrame(index = df.index)

    # Verifica cada feature, transformando somente as features categóricas/object
    for col, col_data in df.iteritems():

        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        output = output.join(col_data)
    
    return output


# In[13]:


# Convertendo as features presentes no dataframes em dummies :

# Datasets balanceados
# Conjunto de treinamento
conjunto1_train = get_dummies_features(conjunto1_train)
conjunto2_train = get_dummies_features(conjunto2_train)
conjunto3_train = get_dummies_features(conjunto3_train)
conjunto4_train = get_dummies_features(conjunto4_train)

# Conjunto de teste
conjunto1_test = get_dummies_features(conjunto1_test)
conjunto2_test = get_dummies_features(conjunto2_test) 
conjunto3_test = get_dummies_features(conjunto3_test)
conjunto4_test = get_dummies_features(conjunto4_test)

# Datasets desbalanceados
conjunto1_train_desbalanceado = get_dummies_features(conjunto1_train_desbalanceado)
conjunto2_train_desbalanceado = get_dummies_features(conjunto2_train_desbalanceado)
conjunto3_train_desbalanceado = get_dummies_features(conjunto3_train_desbalanceado)
conjunto4_train_desbalanceado = get_dummies_features(conjunto4_train_desbalanceado)

# Conjunto de teste
conjunto1_test_desbalanceado = get_dummies_features(conjunto1_test_desbalanceado)
conjunto2_test_desbalanceado = get_dummies_features(conjunto2_test_desbalanceado) 
conjunto3_test_desbalanceado = get_dummies_features(conjunto3_test_desbalanceado)
conjunto4_test_desbalanceado = get_dummies_features(conjunto4_test_desbalanceado)


# In[14]:


conjunto1_train_target = conjunto1_train_target.earnings.replace(-1,0).values
conjunto2_train_target = conjunto2_train_target.earnings.replace(-1,0).values
conjunto3_train_target = conjunto3_train_target.earnings.replace(-1,0).values
conjunto4_train_target = conjunto4_train_target.earnings.replace(-1,0).values
                              
conjunto1_test_target = conjunto1_test_target.earnings.replace(-1,0).values
conjunto2_test_target = conjunto2_test_target.earnings.replace(-1,0).values
conjunto3_test_target = conjunto3_test_target.earnings.replace(-1,0).values
conjunto4_test_target = conjunto4_test_target.earnings.replace(-1,0).values

conjunto1_train_target_desbalanceado = conjunto1_train_target_desbalanceado.earnings.replace(-1,0).values
conjunto2_train_target_desbalanceado = conjunto2_train_target_desbalanceado.earnings.replace(-1,0).values
conjunto3_train_target_desbalanceado = conjunto3_train_target_desbalanceado.earnings.replace(-1,0).values
conjunto4_train_target_desbalanceado = conjunto4_train_target_desbalanceado.earnings.replace(-1,0).values

conjunto1_test_target_desbalanceado = conjunto1_test_target_desbalanceado.earnings.replace(-1,0).values
conjunto2_test_target_desbalanceado = conjunto2_test_target_desbalanceado.earnings.replace(-1,0).values
conjunto3_test_target_desbalanceado = conjunto3_test_target_desbalanceado.earnings.replace(-1,0).values
conjunto4_test_target_desbalanceado = conjunto4_test_target_desbalanceado.earnings.replace(-1,0).values


# In[15]:


def validation_graph_acc(model,X,y,hyperparameter,grid_search,k_folds,score,jobs=1):
    ''' Cria a curva de validação, para busca do hiperparâmetro ótimo'''

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    scores_treino, scores_validacao = validation_curve(model,
                                                       X=X,
                                                       y=y,
                                                       param_name=hyperparameter,
                                                       param_range=grid_search,
                                                       cv=k_folds,
                                                       scoring=score,
                                                       n_jobs=jobs)
    
    # Estatísticas do treino
    train_scores_mean = np.mean(scores_treino   , axis=1)
    train_scores_std  = np.std (scores_treino   , axis=1)
    test_scores_mean  = np.mean(scores_validacao, axis=1)
    test_scores_std   = np.std (scores_validacao, axis=1)

    # Estilo do sns
    sns.set_style('whitegrid')

    # Parâmetros
    param_range = grid_search
    lw = 2

    # Ajuste dos subplots
    plt.figure(figsize=(30,8))
    plt.title('Acurácia para n árvores',fontsize=20)
    plt.xlabel('Quantidade de árvores',fontsize=20)
    plt.ylabel('Acurácia',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)

    plt.plot(param_range, train_scores_mean, label="Acurácia no treinamento",color="darkorange", lw=lw)
    plt.fill_between(param_range, (train_scores_mean - train_scores_std),(train_scores_mean + train_scores_std), alpha=0.2,color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Acurácia na validação cruzada",color="navy", lw=lw)
    plt.fill_between(param_range, (test_scores_mean - test_scores_std),(test_scores_mean + test_scores_std), alpha=0.2,color="navy", lw=lw)
    plt.legend(loc="best",fontsize='xx-large')
    plt.show()


# In[16]:


def validation_graph_roc(model,X,y,hyperparameter,grid_search,k_folds,score,jobs=1):
    ''' Cria a curva de validação, para busca do hiperparâmetro ótimo'''

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    scores_treino, scores_validacao = validation_curve(model,
                                                       X=X,
                                                       y=y,
                                                       param_name=hyperparameter,
                                                       param_range=grid_search,
                                                       cv=k_folds,
                                                       scoring=score,
                                                       n_jobs=jobs)
    
    # Estatísticas do treino
    train_scores_mean = np.mean(scores_treino   , axis=1)
    train_scores_std  = np.std (scores_treino   , axis=1)
    test_scores_mean  = np.mean(scores_validacao, axis=1)
    test_scores_std   = np.std (scores_validacao, axis=1)

    # Estilo do sns
    sns.set_style('whitegrid')

    # Parâmetros
    param_range = grid_search
    lw = 2

    # Ajuste dos subplots
    plt.figure(figsize=(30,8))
    plt.title('ROC AUC para n árvores ',fontsize=20)
    plt.xlabel('Quantidade de árvores',fontsize=20)
    plt.ylabel('ROC AUC',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)

    plt.plot(param_range, train_scores_mean, label="Acurácia no treinamento",color="darkorange", lw=lw)
    plt.fill_between(param_range, (train_scores_mean - train_scores_std),(train_scores_mean + train_scores_std), alpha=0.2,color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="ROC AUC na validação cruzada",color="navy", lw=lw)
    plt.fill_between(param_range, (test_scores_mean - test_scores_std),(test_scores_mean + test_scores_std), alpha=0.2,color="navy", lw=lw)
    plt.legend(loc="best",fontsize='xx-large')
    plt.show()


# In[17]:


def validation_graph_f1(model,X,y,hyperparameter,grid_search,k_folds,score,jobs=1):
    ''' Cria a curva de validação, para busca do hiperparâmetro ótimo'''

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    scores_treino, scores_validacao = validation_curve(model,
                                                       X=X,
                                                       y=y,
                                                       param_name=hyperparameter,
                                                       param_range=grid_search,
                                                       cv=k_folds,
                                                       scoring=score,
                                                       n_jobs=jobs)
    
    # Estatísticas do treino
    train_scores_mean = np.mean(scores_treino   , axis=1)
    train_scores_std  = np.std (scores_treino   , axis=1)
    test_scores_mean  = np.mean(scores_validacao, axis=1)
    test_scores_std   = np.std (scores_validacao, axis=1)

    # Estilo do sns
    sns.set_style('whitegrid')

    # Parâmetros
    param_range = grid_search
    lw = 2

    # Ajuste dos subplots
    plt.figure(figsize=(30,8))
    plt.title('F1-score para n árvores ',fontsize=20)
    plt.xlabel('Quantidade de árvores',fontsize=20)
    plt.ylabel('F1-score',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)

    plt.plot(param_range, train_scores_mean, label="Acurácia no treinamento",color="darkorange", lw=lw)
    plt.fill_between(param_range, (train_scores_mean - train_scores_std),(train_scores_mean + train_scores_std), alpha=0.2,color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Score F1 na validação cruzada",color="navy", lw=lw)
    plt.fill_between(param_range, (test_scores_mean - test_scores_std),(test_scores_mean + test_scores_std), alpha=0.2,color="navy", lw=lw)
    plt.legend(loc="best",fontsize='xx-large')
    plt.show()
    


# ## Redes neurais
# 
# Para a utilização da rede neural como um classificador, devemos definir alguns parâmetros, tais como a quantidade de camadas escondidas, a quantidade de neurônios em cada camada escondida, a quantidade de nós na saída e a função erro a ser utilizada. Os seguintes parâmetros foram escolhidos:
# 
# - Neurônios na saída: sigmoid, 1 neurônio
# - Função de ativação: ReLU
# - Função erro: Entropia cruzada
# 
# O treinamento da rede neural é feito utilizando o pacote Keras em conjunto com o tensorflow.
# 

# In[18]:


# Importando o keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation


# In[19]:


# Exemplo de rede - teste básico
# Criação de uma rede
model = Sequential()
model.add(Dense(100, input_shape=(37,), activation='relu'))
#model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Problema de classificação binária,
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[25]:


# Treinamento da rede
model.fit(conjunto1_train.values,conjunto1_train_target, epochs=3000, batch_size=50)


# ### Cenários para os datasets balanceados

# ### Dataset 1

# ### Dataset 2

# ### Dataset 3

# ### Dataset 4
# 
# 

# ### Avaliação dos modelos para cada cenário

# ### Cenários para os datasets desbalanceado

# ### Dataset 1

# #### F1-Score

# #### ROC-AUC

# ### Dataset 2

# #### F1-Score

# #### ROC-AUC

# ### Dataset 3

# #### F1-Score

# #### ROC-AUC

# ### Dataset 4

# #### F1-Score

# #### ROC-AUC

# ### Avaliação dos modelos para cada cenário
