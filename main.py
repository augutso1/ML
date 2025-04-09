import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Função para carregar e preparar os dados do dataset
# - Carrega o arquivo CSV especificado
# - Seleciona apenas as colunas relevantes (preço, área, quartos, andares)
def carregar_dataset(caminho):
    df = pd.read_csv(caminho)  # Carrega o dataset do arquivo CSV
    # Garantir que todas as colunas categóricas sejam do tipo "object" ou "string"
    return df

# Definição da arquitetura da rede neural para previsão de preços de imóveis
# - Usa três camadas lineares (fully connected)
# - Utiliza ReLU como função de ativação
class HousePricePredictor(nn.Module):
    def __init__(self, input_size):
        super(HousePricePredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x

# Função principal que organiza todo o fluxo de execução
def main():
    # Carregamento e preparação dos dados
    df = carregar_dataset('Housing.csv')  # Carrega o dataset de preços de imóveis
    
    # Identificar colunas numéricas e categóricas
    colunas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    colunas_numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    colunas_numericas = [col for col in colunas_numericas if col != 'price']
    
    # Converter colunas categóricas para numéricas (0/1)
    for col in colunas_categoricas:
        # Verificar se é uma coluna binária (yes/no)
        if set(df[col].unique()) == {'yes', 'no'} or set(df[col].unique()) == {'no', 'yes'}:
            df[col] = df[col].map({'yes': 1, 'no': 0})
        else:
            # Para colunas não binárias, criar variáveis dummy (one-hot encoding)
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)
    
    # Atualizar lista de colunas numéricas após a conversão
    colunas_features = df.columns.tolist()
    colunas_features.remove('price')
    
    # Separação dos dados em features (X) e target (y)
    X = df[colunas_features].values  # Todas as features (agora numéricas)
    y = df['price'].values.reshape(-1, 1)  # Target: preço
    
    # Normalização dos dados para melhorar a convergência do modelo
    scaler_X = StandardScaler()  # Inicializa o scaler para as features
    scaler_y = StandardScaler()  # Inicializa o scaler para o target
    X_normalized = scaler_X.fit_transform(X)  # Normaliza as features (média 0, desvio padrão 1)
    y_normalized = scaler_y.fit_transform(y)  # Normaliza os preços (média 0, desvio padrão 1)
    
    # Conversão dos dados normalizados para tensores PyTorch
    X_tensor = torch.FloatTensor(X_normalized)  # Converte features para tensor
    y_tensor = torch.FloatTensor(y_normalized)  # Converte preços para tensor
    
    # Obter o número final de features após one-hot encoding
    input_size = X.shape[1]
    
    # Inicialização e configuração do modelo
    model = HousePricePredictor(input_size=input_size)  # Cria modelo com o número correto de features
    criterion = nn.MSELoss()  # Define função de perda como Erro Quadrático Médio (MSE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define otimizador Adam com taxa de aprendizagem 0.01
    
    # Loop de treinamento do modelo
    epochs = 100  # Número total de epochs para treinar
    for epoch in range(epochs):
        # Forward pass: calcula previsões e perda
        outputs = model(X_tensor)  # Passa dados pelo modelo para obter previsões
        loss = criterion(outputs, y_tensor)  # Calcula erro entre previsões e valores reais
        
        # Backward pass: atualiza os pesos do modelo
        optimizer.zero_grad()  # Zera gradientes acumulados
        loss.backward()        # Calcula gradientes via backpropagation
        optimizer.step()       # Atualiza pesos do modelo baseado nos gradientes
        
        # Exibe progresso a cada 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    # Criar exemplo de teste
    teste_dict = {}
    teste_dict['area'] = 7420
    teste_dict['bedrooms'] = 4
    teste_dict['bathrooms'] = 2
    teste_dict['stories'] = 3
    
    # Preencher valores binários (0 ou 1)
    for col in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea']:
        if col in colunas_features:
            teste_dict[col] = 1  # Assumindo 'yes' para todas como exemplo
    
    # Preencher valores para colunas one-hot encoded
    if 'furnishingstatus' in colunas_categoricas:
        for col in colunas_features:
            if col.startswith('furnishingstatus_'):
                teste_dict[col] = 1 if col == 'furnishingstatus_furnished' else 0
    
    # Criar array de teste com os valores correspondentes às features em X
    teste = np.array([[teste_dict.get(col, 0) for col in colunas_features]])
    print("Exemplo de teste:", teste)
    
    # Normaliza o exemplo de teste
    teste_normalized = scaler_X.transform(teste)  # Normaliza o exemplo de teste
    teste_tensor = torch.FloatTensor(teste_normalized)  # Converte para tensor
    
    # Realiza a previsão
    model.eval()  # Coloca o modelo em modo de avaliação
    with torch.no_grad():  # Desativa cálculo de gradientes para inferência
        predicao = model(teste_tensor)  # Obtém previsão do modelo
        predicao = scaler_y.inverse_transform(predicao.numpy())  # Converte de volta para escala original
    
    # Exibe o resultado da previsão
    print(f'\nPrevisão de preço: ${predicao[0][0]:.2f}')

if __name__ == "__main__":
    main()  # Executa a função principal quando o script é rodado diretamente