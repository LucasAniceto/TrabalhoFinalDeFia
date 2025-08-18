# Makefile para automação do projeto ML E-commerce

.PHONY: help install test lint format run clean deploy

help:  ## Mostra esta mensagem de ajuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install:  ## Instala dependências
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 bandit

test:  ## Executa testes unitários
	pytest test_ml_ecommerce.py -v --cov=ml_ecommerce --cov-report=html --cov-report=term

test-ci:  ## Executa testes para CI/CD
	pytest test_ml_ecommerce.py -v --cov=ml_ecommerce --cov-report=xml

lint:  ## Verifica qualidade do código
	flake8 ml_ecommerce.py test_ml_ecommerce.py --max-line-length=127
	bandit -r . -f txt

format:  ## Formata código com black
	black ml_ecommerce.py test_ml_ecommerce.py

format-check:  ## Verifica formatação sem aplicar
	black --check --diff ml_ecommerce.py test_ml_ecommerce.py

run:  ## Executa o modelo principal
	python ml_ecommerce.py

run-monitoring:  ## Executa monitoramento de drift
	python -c "from ml_ecommerce import carregar_dados_referencia, monitorar_drift_dados; print('🔍 Monitoramento iniciado...')"

train-model:  ## Retreina o modelo
	python -c "from ml_ecommerce import main; main()"

validate-model:  ## Valida performance do modelo
	python -c "
	import numpy as np, pandas as pd
	from ml_ecommerce import ModeloRetreinamento
	retreinamento = ModeloRetreinamento()
	modelo_data = retreinamento.carregar_modelo()
	if modelo_data:
		print(f'✅ Modelo carregado: {modelo_data[\"timestamp\"]}')
		print(f'📊 Métricas: {modelo_data[\"metadata\"].get(\"metricas\", \"N/A\")}')
	else:
		print('❌ Nenhum modelo encontrado')
	"

clean:  ## Remove arquivos temporários
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -f coverage.xml
	rm -f *.log
	rm -f bandit-report.json

setup-dev:  ## Configura ambiente de desenvolvimento
	python -m venv venv
	@echo "🔧 Ambiente virtual criado. Ative com:"
	@echo "   source venv/bin/activate  # Linux/Mac"
	@echo "   venv\\Scripts\\activate     # Windows"
	@echo "🔧 Depois execute: make install"

check-deps:  ## Verifica dependências
	pip check
	pip list --outdated

security-scan:  ## Executa scan de segurança
	bandit -r . -f json -o bandit-report.json
	@echo "📋 Relatório salvo em: bandit-report.json"

generate-docs:  ## Gera documentação
	@echo "📚 Gerando documentação..."
	python -c "
	from ml_ecommerce import *
	import inspect
	
	print('# Documentação das Funções\\n')
	
	functions = [
		carregar_dados, limpar_dados, engenharia_features,
		treinar_modelos_multiplos, analisar_produtos,
		criar_visualizacoes, monitorar_drift_dados
	]
	
	for func in functions:
		print(f'## {func.__name__}')
		doc = inspect.getdoc(func)
		if doc:
			print(f'{doc}\\n')
		else:
			print('Sem documentação disponível\\n')
	" > DOCS.md
	@echo "📋 Documentação salva em: DOCS.md"

deploy-local:  ## Deploy local para testes
	@echo "🚀 Preparando deploy local..."
	mkdir -p deploy
	cp ml_ecommerce.py requirements.txt deploy/
	cp *.html deploy/ 2>/dev/null || echo "📊 Nenhum HTML encontrado"
	@echo "✅ Deploy preparado em ./deploy/"

benchmark:  ## Executa benchmark de performance
	python -c "
	import time
	import numpy as np
	import pandas as pd
	from ml_ecommerce import treinar_modelos_multiplos, limpar_dados, engenharia_features
	
	print('⏱️ Executando benchmark...')
	
	# Gerar dados de diferentes tamanhos
	sizes = [100, 500, 1000, 2000]
	
	for size in sizes:
		np.random.seed(42)
		df_test = pd.DataFrame({
			'price': np.random.lognormal(3, 0.5, size),
			'freight_value': np.random.normal(10, 3, size),
			'product_weight_g': np.random.lognormal(6, 1, size),
			'product_length_cm': np.random.normal(20, 5, size),
			'product_height_cm': np.random.normal(10, 3, size),
			'product_width_cm': np.random.normal(15, 4, size),
			'order_status': ['delivered'] * size,
			'product_id': [f'prod_{i}' for i in range(size)],
			'order_purchase_timestamp': pd.date_range('2023-01-01', periods=size, freq='H'),
			'customer_state': np.random.choice(['SP', 'RJ', 'MG'], size),
			'seller_state': np.random.choice(['SP', 'RJ', 'MG'], size),
			'review_score': np.random.uniform(1, 5, size),
			'num_reviews': np.random.poisson(5, size),
			'product_category_name': np.random.choice(['electronics', 'books', 'fashion'], size),
			'seller_id': [f'seller_{i%50}' for i in range(size)]
		})
		
		start_time = time.time()
		try:
			df_clean = limpar_dados(df_test)
			df_features, le1, le2, le3 = engenharia_features(df_clean)
			resultado = treinar_modelos_multiplos(df_features)
			
			elapsed = time.time() - start_time
			print(f'📊 {size:4d} registros: {elapsed:.2f}s')
		except Exception as e:
			print(f'❌ {size:4d} registros: Erro - {e}')
	"

monitor-drift:  ## Monitora drift em dados salvos
	python -c "
	from ml_ecommerce import carregar_dados_referencia, monitorar_drift_dados
	import pandas as pd
	import numpy as np
	
	print('🔍 Verificando drift nos dados...')
	
	dados_ref = carregar_dados_referencia()
	if dados_ref is not None:
		# Simular novos dados com drift
		dados_novos = dados_ref.copy()
		# Adicionar drift artificial em algumas colunas
		if 'price' in dados_novos.columns:
			dados_novos['price'] = dados_novos['price'] * 1.1  # 10% de aumento
		
		resultado_drift = monitorar_drift_dados(dados_ref, dados_novos)
		print(f'📈 Score de drift: {resultado_drift[\"drift_score\"]:.1f}%')
		print(f'🎯 Recomendação: {resultado_drift[\"recomendacao\"]}')
	else:
		print('❌ Dados de referência não encontrados. Execute o modelo primeiro.')
	"

all: format lint test  ## Executa formatação, lint e testes

# Meta-target para CI/CD
ci: format-check lint test-ci security-scan  ## Pipeline completo de CI/CD