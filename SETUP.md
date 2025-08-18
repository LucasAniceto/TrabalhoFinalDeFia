# Setup do Sistema de Recomendação de Preços

## Instalação

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Configuração do Kaggle (Recomendado)

O sistema agora baixa automaticamente os dados do Kaggle. Para configurar:

1. Crie uma conta no [Kaggle](https://www.kaggle.com)

2. Vá em "Account" > "API" e clique em "Create New Token"

3. Isso baixará um arquivo `kaggle.json` com suas credenciais

4. Configure as credenciais do Kaggle:
   - **Linux/Mac**: Coloque o arquivo em `~/.kaggle/kaggle.json`
   - **Windows**: Coloque o arquivo em `C:\Users\<username>\.kaggle\kaggle.json`

5. Defina as permissões do arquivo (Linux/Mac):
```bash
chmod 600 ~/.kaggle/kaggle.json
```

## Execução

```bash
python ml_ecommerce.py
```

O sistema tentará primeiro baixar os dados do Kaggle. Se falhar, você pode usar arquivos locais.

## Uso com Arquivos Locais

Se preferir usar arquivos locais, modifique a função main() no código:

```python
# Em vez de:
datasets = carregar_dados(usar_kaggle=True)

# Use:
datasets = carregar_dados(caminho_base="caminho/para/seus/arquivos", usar_kaggle=False)
```

## Dataset

Este projeto usa o dataset "Brazilian E-Commerce Public Dataset by Olist" disponível em:
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce