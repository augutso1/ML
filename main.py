import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

# Criar diretórios para templates e arquivos estáticos
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Criar o arquivo HTML
with open("templates/index.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Previsão de Preço de Imóveis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
    <div class="container mt-5">
        <div class="row justify-content-center">
            <div class="col-md-8">
                <div class="card shadow">
                    <div class="card-header bg-primary text-white">
                        <h3 class="mb-0">Previsão de Preço de Imóveis</h3>
                    </div>
                    <div class="card-body">
                        <form method="post" action="/predict">
                            <div class="row">
                                <div class="col-md-6 mb-3">
                                    <label for="area" class="form-label">Área (pés²)</label>
                                    <input type="number" class="form-control" id="area" name="area" required>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <label for="bedrooms" class="form-label">Quartos</label>
                                    <input type="number" class="form-control" id="bedrooms" name="bedrooms" required>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <label for="bathrooms" class="form-label">Banheiros</label>
                                    <input type="number" class="form-control" id="bathrooms" name="bathrooms" required>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <label for="stories" class="form-label">Andares</label>
                                    <input type="number" class="form-control" id="stories" name="stories" required>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <label for="mainroad" class="form-label">Acesso à Rua Principal</label>
                                    <select class="form-select" id="mainroad" name="mainroad" required>
                                        <option value="1">Sim</option>
                                        <option value="0">Não</option>
                                    </select>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <label for="guestroom" class="form-label">Quarto de Hóspedes</label>
                                    <select class="form-select" id="guestroom" name="guestroom" required>
                                        <option value="1">Sim</option>
                                        <option value="0">Não</option>
                                    </select>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <label for="basement" class="form-label">Porão</label>
                                    <select class="form-select" id="basement" name="basement" required>
                                        <option value="1">Sim</option>
                                        <option value="0">Não</option>
                                    </select>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <label for="hotwaterheating" class="form-label">Aquecimento de Água</label>
                                    <select class="form-select" id="hotwaterheating" name="hotwaterheating" required>
                                        <option value="1">Sim</option>
                                        <option value="0">Não</option>
                                    </select>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <label for="airconditioning" class="form-label">Ar Condicionado</label>
                                    <select class="form-select" id="airconditioning" name="airconditioning" required>
                                        <option value="1">Sim</option>
                                        <option value="0">Não</option>
                                    </select>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <label for="parking" class="form-label">Vagas de Garagem</label>
                                    <input type="number" class="form-control" id="parking" name="parking" required>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <label for="prefarea" class="form-label">Área Preferencial</label>
                                    <select class="form-select" id="prefarea" name="prefarea" required>
                                        <option value="1">Sim</option>
                                        <option value="0">Não</option>
                                    </select>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <label for="furnishingstatus" class="form-label">Mobília</label>
                                    <select class="form-select" id="furnishingstatus" name="furnishingstatus" required>
                                        <option value="furnished">Mobiliado</option>
                                        <option value="semi-furnished">Semi-mobiliado</option>
                                        <option value="unfurnished">Não mobiliado</option>
                                    </select>
                                </div>
                            </div>
                            <div class="d-grid gap-2">
                                <button type="submit" class="btn btn-primary">Calcular Preço</button>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    """)

# Criar a aplicação FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Variáveis globais para armazenar o modelo e os scalers
model = None
scaler_X = None
scaler_y = None
colunas_features = None

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

# Função para treinar o modelo
def treinar_modelo():
    global model, scaler_X, scaler_y, colunas_features
    
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
    
    # Divisão dos dados em conjuntos de treino, validação e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 60% treino, 20% val, 20% teste
    
    # Normalização dos dados para melhorar a convergência do modelo
    scaler_X = StandardScaler()  # Inicializa o scaler para as features
    scaler_y = StandardScaler()  # Inicializa o scaler para o target
    X_train_normalized = scaler_X.fit_transform(X_train)  # Normaliza as features (média 0, desvio padrão 1)
    y_train_normalized = scaler_y.fit_transform(y_train)  # Normaliza os preços (média 0, desvio padrão 1)
    
    # Normalizar conjuntos de validação e teste usando os mesmos scalers
    X_val_normalized = scaler_X.transform(X_val)
    y_val_normalized = scaler_y.transform(y_val)
    X_test_normalized = scaler_X.transform(X_test)
    y_test_normalized = scaler_y.transform(y_test)
    
    # Conversão dos dados normalizados para tensores PyTorch
    X_train_tensor = torch.FloatTensor(X_train_normalized)
    y_train_tensor = torch.FloatTensor(y_train_normalized)
    X_val_tensor = torch.FloatTensor(X_val_normalized)
    y_val_tensor = torch.FloatTensor(y_val_normalized)
    X_test_tensor = torch.FloatTensor(X_test_normalized)
    y_test_tensor = torch.FloatTensor(y_test_normalized)
    
    # Criar DataLoaders para processar os dados em lotes
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Obter o número final de features após one-hot encoding
    input_size = X.shape[1]
    
    # Inicialização e configuração do modelo
    model = HousePricePredictor(input_size=input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Configurações de treinamento
    epochs = 100
    best_val_loss = float('inf')
    patience = 10
    counter = 0
    
    # Loop de treinamento do modelo com early stopping
    for epoch in range(epochs):
        # Modo de treinamento
        model.train()
        train_loss = 0
        
        # Processamento em lotes
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calcular perda média por época
        train_loss /= len(train_loader)
        
        # Modo de avaliação para validação
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_outputs = model(val_X)
                val_batch_loss = criterion(val_outputs, val_y)
                val_loss += val_batch_loss.item()
        
        val_loss /= len(val_loader)
        
        # Mostrar progresso a cada época
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Salvar o melhor modelo
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping após {epoch+1} epochs')
                break
    
    # Carregar o melhor modelo
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()  # Colocar o modelo em modo de avaliação

# Função para fazer previsão com o modelo
def fazer_previsao(dados):
    global model, scaler_X, scaler_y, colunas_features
    
    # Criar um DataFrame com os dados recebidos
    df_teste = pd.DataFrame([dados])
    
    # Garantir que todas as colunas necessárias estejam presentes
    for col in colunas_features:
        if col not in df_teste.columns:
            df_teste[col] = 0
    
    # Reordenar as colunas para corresponder ao treinamento
    df_teste = df_teste[colunas_features]
    
    # Converter para array numpy
    X_teste = df_teste.values
    
    # Normalizar os dados
    X_teste_normalized = scaler_X.transform(X_teste)
    
    # Converter para tensor
    X_teste_tensor = torch.FloatTensor(X_teste_normalized)
    
    # Fazer a previsão
    with torch.no_grad():
        predicao = model(X_teste_tensor)
        predicao = scaler_y.inverse_transform(predicao.numpy())
    
    return predicao[0][0]

# Rota para a página inicial
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Rota para receber os dados e fazer a previsão
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    area: float = Form(...),
    bedrooms: int = Form(...),
    bathrooms: int = Form(...),
    stories: int = Form(...),
    mainroad: int = Form(...),
    guestroom: int = Form(...),
    basement: int = Form(...),
    hotwaterheating: int = Form(...),
    airconditioning: int = Form(...),
    parking: int = Form(...),
    prefarea: int = Form(...),
    furnishingstatus: str = Form(...)
):
    # Criar dicionário com os dados recebidos
    dados = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'mainroad': mainroad,
        'guestroom': guestroom,
        'basement': basement,
        'hotwaterheating': hotwaterheating,
        'airconditioning': airconditioning,
        'parking': parking,
        'prefarea': prefarea
    }
    
    # Adicionar variáveis dummy para furnishingstatus
    dados[f'furnishingstatus_{furnishingstatus}'] = 1
    
    # Fazer a previsão
    preco_previsto = fazer_previsao(dados)
    
    # Criar HTML com o resultado
    html_resultado = f"""
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Resultado da Previsão</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
        <div class="container mt-5">
            <div class="row justify-content-center">
                <div class="col-md-8">
                    <div class="card shadow">
                        <div class="card-header bg-success text-white">
                            <h3 class="mb-0">Resultado da Previsão</h3>
                        </div>
                        <div class="card-body">
                            <div class="alert alert-success">
                                <h4 class="alert-heading">Preço Previsto:</h4>
                                <p class="display-4">${preco_previsto:.2f}</p>
                            </div>
                            <div class="d-grid gap-2">
                                <a href="/" class="btn btn-primary">Voltar</a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_resultado)

# Função principal que organiza todo o fluxo de execução
def main():
    # Treinar o modelo
    treinar_modelo()
    
    # Iniciar o servidor FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()  # Executa a função principal quando o script é rodado diretamente