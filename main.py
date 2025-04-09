import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Função para carregar e preparar os dados do dataset
# - Carrega o arquivo CSV especificado
# - Seleciona apenas as colunas relevantes (preço, área, quartos, andares)
def carregar_dataset(caminho):
    df = pd.read_csv(caminho)  # Carrega o dataset do arquivo CSV
    df = df[['price','area', 'bedrooms','stories']]  # Filtra apenas as colunas necessárias
    return df

# Definição da arquitetura da rede neural para previsão de preços de imóveis
# - Usa três camadas lineares (fully connected)
# - Utiliza ReLU como função de ativação
class HousePricePredictor(nn.Module):
    def __init__(self, input_size):
        super(HousePricePredictor, self).__init__()  # Inicializa a classe base
        self.layer1 = nn.Linear(input_size, 64)      # Primeira camada: entrada -> 64 neurônios
        self.layer2 = nn.Linear(64, 32)              # Segunda camada: 64 -> 32 neurônios
        self.layer3 = nn.Linear(32, 1)               # Camada de saída: 32 -> 1 neurônio (preço)
        self.relu = nn.ReLU()                        # Função de ativação ReLU
        
    # Método que define o fluxo de dados através da rede
    def forward(self, x):
        x = self.relu(self.layer1(x))  # Aplica primeira camada + ativação ReLU
        x = self.relu(self.layer2(x))  # Aplica segunda camada + ativação ReLU
        x = self.layer3(x)             # Aplica camada de saída (sem ativação)
        return x

# Função principal que organiza todo o fluxo de execução
def main():
    # Carregamento e preparação dos dados
    df = carregar_dataset('Housing.csv')  # Carrega o dataset de preços de imóveis
    print("Shape de X:", df[['area', 'bedrooms', 'stories']].values.shape)  # Exibe dimensões dos dados para debug
    
    # Separação dos dados em features (X) e target (y)
    X = df[['area', 'bedrooms', 'stories']].values  # Features: área, quartos e andares
    y = df['price'].values.reshape(-1, 1)           # Target: preço (reshape para formato de coluna)
    
    # Normalização dos dados para melhorar a convergência do modelo
    scaler_X = StandardScaler()  # Inicializa o scaler para as features
    scaler_y = StandardScaler()  # Inicializa o scaler para o target
    X_normalized = scaler_X.fit_transform(X)  # Normaliza as features (média 0, desvio padrão 1)
    y_normalized = scaler_y.fit_transform(y)  # Normaliza os preços (média 0, desvio padrão 1)
    
    # Conversão dos dados normalizados para tensores PyTorch
    X_tensor = torch.FloatTensor(X_normalized)  # Converte features para tensor
    y_tensor = torch.FloatTensor(y_normalized)  # Converte preços para tensor
    
    # Inicialização e configuração do modelo
    model = HousePricePredictor(input_size=3)  # Cria modelo com 3 features de entrada
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
    
    # Teste do modelo em um novo exemplo
    teste = np.array([[7420, 4, 20]])  # Exemplo: imóvel com 7420 pés², 4 quartos, 20 anos
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