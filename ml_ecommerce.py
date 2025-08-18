# ============================================================================
# SISTEMA DE RECOMENDAÇÃO DE PREÇOS PARA E-COMMERCE - VERSÃO CORRIGIDA
# Dataset: Olist Brazilian E-commerce (Local)
# Algoritmo: Random Forest Regressor
# ============================================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFECV
import xgboost as xgb
from scipy import stats
import logging
import warnings
import os
import json
import pickle
import schedule
import time
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional, Any
import hashlib

# Importações condicionais para features avançadas
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP não disponível. Instale com: pip install shap")

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_suite import MetricSuite
    from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesMetric
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("⚠️ Evidently não disponível. Instale com: pip install evidently")

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_ecommerce.log'),
        logging.StreamHandler()
    ]
)

# Configurações de visualização
# Plotly não requer configurações globais específicas

# Cores para output colorido
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'

def print_header(text):
    print(f"\n{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text:^80}{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")

def print_section(text):
    print(f"\n{Colors.YELLOW}{Colors.BOLD}🔄 {text}{Colors.RESET}")
    print(f"{Colors.BLUE}{'─'*60}{Colors.RESET}")

# ============================================================================
# 1. CARREGAMENTO E EXPLORAÇÃO DOS DADOS
# ============================================================================

def carregar_dados(caminho_base):
    """Carrega todos os datasets do Olist"""
    print_section("CARREGANDO DATASETS")
    
    try:
        arquivos = {
            'orders': 'olist_orders_dataset.csv',
            'order_items': 'olist_order_items_dataset.csv',
            'products': 'olist_products_dataset.csv',
            'sellers': 'olist_sellers_dataset.csv',
            'customers': 'olist_customers_dataset.csv',
            'reviews': 'olist_order_reviews_dataset.csv',
            'category_translation': 'product_category_name_translation.csv'
        }
        
        datasets = {}
        
        for nome, arquivo in arquivos.items():
            caminho_completo = os.path.join(caminho_base, arquivo)
            if os.path.exists(caminho_completo):
                datasets[nome] = pd.read_csv(caminho_completo)
                print(f"   ✅ {nome}: {datasets[nome].shape} - {arquivo}")
            else:
                print(f"   ❌ {nome}: Arquivo não encontrado - {arquivo}")
        
        return datasets
    
    except Exception as e:
        print(f"{Colors.RED}❌ Erro ao carregar dados: {e}{Colors.RESET}")
        return None

def juntar_datasets(datasets):
    """Junta todos os datasets em um dataset principal"""
    print_section("JUNTANDO DATASETS")
    
    df_main = datasets['order_items'].copy()
    print(f"   📊 Base (order_items): {df_main.shape}")
    
    if 'products' in datasets:
        df_main = df_main.merge(datasets['products'], on='product_id', how='left')
        print(f"   📊 + products: {df_main.shape}")
    
    if 'category_translation' in datasets:
        df_main = df_main.merge(datasets['category_translation'], 
                               on='product_category_name', how='left')
        print(f"   📊 + category_translation: {df_main.shape}")
    
    if 'orders' in datasets:
        df_main = df_main.merge(datasets['orders'][['order_id', 'customer_id', 'order_status',
                                                   'order_purchase_timestamp']], 
                               on='order_id', how='left')
        print(f"   📊 + orders: {df_main.shape}")
    
    if 'customers' in datasets:
        df_main = df_main.merge(datasets['customers'][['customer_id', 'customer_city',
                                                      'customer_state']], 
                               on='customer_id', how='left')
        print(f"   📊 + customers: {df_main.shape}")
    
    if 'sellers' in datasets:
        df_main = df_main.merge(datasets['sellers'][['seller_id', 'seller_city', 
                                                    'seller_state']], 
                               on='seller_id', how='left')
        print(f"   📊 + sellers: {df_main.shape}")
    
    if 'reviews' in datasets:
        reviews_agg = datasets['reviews'].groupby('order_id').agg({
            'review_score': 'mean',
            'review_id': 'count'
        }).rename(columns={'review_id': 'num_reviews'}).reset_index()
        
        df_main = df_main.merge(reviews_agg, on='order_id', how='left')
        print(f"   📊 + reviews (agregadas): {df_main.shape}")
    
    print(f"\n   🎯 Dataset final: {df_main.shape}")
    return df_main

# ============================================================================
# 2. LIMPEZA E PREPARAÇÃO DOS DADOS
# ============================================================================

def detectar_outliers_avancado(df, metodo='isolation_forest'):
    """Detecta outliers usando métodos estatísticos avançados"""
    logging.info(f"Detectando outliers usando método: {metodo}")
    
    if metodo == 'isolation_forest':
        # Usar Isolation Forest para detectar outliers multivariados
        features_numericas = ['price', 'freight_value', 'product_weight_g', 
                             'product_length_cm', 'product_height_cm', 'product_width_cm']
        features_disponiveis = [f for f in features_numericas if f in df.columns]
        
        if len(features_disponiveis) > 0:
            X_outliers = df[features_disponiveis].fillna(df[features_disponiveis].median())
            
            iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
            outliers_pred = iso_forest.fit_predict(X_outliers)
            
            # -1 indica outlier, 1 indica normal
            outliers_mask = outliers_pred == 1
            
            print(f"   🔍 Isolation Forest removeu {len(df) - outliers_mask.sum()} outliers")
            return df[outliers_mask].copy()
    
    # Fallback para método estatístico tradicional
    Q1 = df['price'].quantile(0.01)
    Q3 = df['price'].quantile(0.99)
    df_filtered = df[(df['price'] >= Q1) & (df['price'] <= Q3)]
    print(f"   📊 Método estatístico removeu {len(df) - len(df_filtered)} outliers")
    
    return df_filtered

def limpar_dados(df):
    """Limpa e prepara os dados para análise com métodos avançados"""
    print_section("LIMPEZA E PREPARAÇÃO DOS DADOS")
    logging.info("Iniciando limpeza de dados")
    
    df_clean = df.copy()
    
    # Filtrar apenas pedidos entregues
    df_clean = df_clean[df_clean['order_status'] == 'delivered'].copy()
    print(f"   📦 Apenas pedidos entregues: {df_clean.shape}")
    logging.info(f"Filtrados pedidos entregues: {df_clean.shape}")
    
    # Remover valores faltantes críticos
    colunas_criticas = ['price', 'product_id']
    df_clean = df_clean.dropna(subset=colunas_criticas)
    print(f"   🧹 Após remover NAs críticos: {df_clean.shape}")
    
    # Detectar outliers usando método avançado
    df_clean = detectar_outliers_avancado(df_clean, metodo='isolation_forest')
    print(f"   📊 Após detecção avançada de outliers: {df_clean.shape}")
    
    # Preencher valores faltantes com estratégias mais sofisticadas
    colunas_numericas = ['review_score', 'num_reviews', 'product_weight_g', 
                        'product_length_cm', 'product_height_cm', 'product_width_cm']
    
    for coluna in colunas_numericas:
        if coluna in df_clean.columns:
            if coluna == 'num_reviews':
                df_clean[coluna].fillna(0, inplace=True)
            elif coluna == 'review_score':
                # Usar mediana por categoria quando possível
                if 'product_category_name' in df_clean.columns:
                    df_clean[coluna] = df_clean.groupby('product_category_name')[coluna].transform(
                        lambda x: x.fillna(x.median()))
                df_clean[coluna].fillna(df_clean[coluna].median(), inplace=True)
            else:
                df_clean[coluna].fillna(df_clean[coluna].median(), inplace=True)
    
    print(f"   ✅ Limpeza concluída: {df_clean.shape}")
    logging.info(f"Limpeza concluída. Shape final: {df_clean.shape}")
    return df_clean

def engenharia_features(df):
    """Cria novas features para melhorar o modelo"""
    print_section("ENGENHARIA DE FEATURES")
    
    df_features = df.copy()
    
    # Features temporais
    df_features['order_purchase_timestamp'] = pd.to_datetime(df_features['order_purchase_timestamp'])
    df_features['order_month'] = df_features['order_purchase_timestamp'].dt.month
    df_features['order_dayofweek'] = df_features['order_purchase_timestamp'].dt.dayofweek
    df_features['order_hour'] = df_features['order_purchase_timestamp'].dt.hour
    df_features['is_weekend'] = (df_features['order_dayofweek'] >= 5).astype(int)
    
    # Features de dimensões do produto
    df_features['product_volume'] = (df_features['product_length_cm'] * 
                                   df_features['product_height_cm'] * 
                                   df_features['product_width_cm'])
    df_features['product_volume'] = df_features['product_volume'].fillna(
        df_features['product_volume'].median())
    
    # Features de densidade
    df_features['density'] = df_features['product_weight_g'] / (df_features['product_volume'] + 1)
    
    # Features de localização
    df_features['same_state'] = (df_features['customer_state'] == 
                               df_features['seller_state']).astype(int)
    
    # Features de qualidade/avaliação
    df_features['quality_score'] = (df_features['review_score'] * 
                                  np.log1p(df_features['num_reviews']))
    
    # Features cíclicas para mês
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['order_month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['order_month'] / 12)
    
    # Encoding de variáveis categóricas
    le_category = LabelEncoder()
    le_customer_state = LabelEncoder()
    le_seller_state = LabelEncoder()
    
    # Preencher valores faltantes antes do encoding
    df_features['product_category_name'].fillna('unknown', inplace=True)
    df_features['customer_state'].fillna('unknown', inplace=True)
    df_features['seller_state'].fillna('unknown', inplace=True)
    
    df_features['category_encoded'] = le_category.fit_transform(df_features['product_category_name'])
    df_features['customer_state_encoded'] = le_customer_state.fit_transform(df_features['customer_state'])
    df_features['seller_state_encoded'] = le_seller_state.fit_transform(df_features['seller_state'])
    
    # Estatísticas por produto
    product_stats = df_features.groupby('product_category_name').agg({
        'price': ['mean', 'std', 'count'],
        'review_score': 'mean'
    }).round(2)
    
    product_stats.columns = ['category_price_mean', 'category_price_std', 
                           'category_count', 'category_review_mean']
    product_stats = product_stats.reset_index()
    
    df_features = df_features.merge(product_stats, on='product_category_name', how='left')
    
    print(f"   ✅ Features criadas. Shape final: {df_features.shape}")
    
    return df_features, le_category, le_customer_state, le_seller_state

# ============================================================================
# 3. MODELAGEM AVANÇADA COM MÚLTIPLOS ALGORITMOS
# ============================================================================

def selecionar_features_shap(modelo, X_sample, y_sample, feature_names, max_features=15):
    """Seleciona features usando importância SHAP"""
    if not SHAP_AVAILABLE:
        logging.warning("SHAP não disponível. Usando importância padrão do modelo.")
        return None, None
    
    try:
        print(f"   🔍 Calculando importância SHAP (amostra de {len(X_sample)} registros)...")
        
        # Criar explainer baseado no tipo de modelo
        if hasattr(modelo, 'estimators_'):  # Random Forest
            explainer = shap.TreeExplainer(modelo)
        else:  # XGBoost ou outros
            explainer = shap.Explainer(modelo)
        
        # Calcular SHAP values para uma amostra menor (para performance)
        sample_size = min(100, len(X_sample))  # Reduzindo para 100 amostras
        if hasattr(X_sample, 'sample'):  # Se for DataFrame
            X_shap_sample = X_sample.sample(n=sample_size, random_state=42).values
        else:  # Se for array numpy
            sample_indices = np.random.choice(len(X_sample), sample_size, replace=False)
            X_shap_sample = X_sample[sample_indices]
        
        print(f"   🔍 Amostra SHAP: {X_shap_sample.shape}")
        
        shap_values = explainer.shap_values(X_shap_sample)
        
        # Calcular importância média absoluta
        if isinstance(shap_values, list):  # Para alguns modelos multi-class
            shap_values = shap_values[0]
        
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        # Criar DataFrame com importâncias
        shap_importance = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Importance': feature_importance
        }).sort_values('SHAP_Importance', ascending=False)
        
        # Selecionar top features
        top_features = shap_importance.head(max_features)['Feature'].tolist()
        features_mask = [f in top_features for f in feature_names]
        
        print(f"   🎯 SHAP selecionou {len(top_features)} features")
        print(f"   📊 Top 5 features SHAP:")
        for i, row in shap_importance.head(5).iterrows():
            print(f"      {row['Feature']}: {row['SHAP_Importance']:.4f}")
        
        return features_mask, shap_importance
    
    except Exception as e:
        logging.error(f"Erro no cálculo SHAP: {e}")
        return None, None

def selecionar_features_automatico(X, y, metodo='rfe'):
    """Seleciona features automaticamente usando diferentes métodos"""
    logging.info(f"Selecionando features usando método: {metodo}")
    
    if metodo == 'shap' and SHAP_AVAILABLE:
        # Treinar modelo temporário para SHAP
        rf_temp = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_temp.fit(X, y)
        
        # Usar nomes de features genéricos se não fornecidos
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        features_mask, shap_importance = selecionar_features_shap(
            rf_temp, X, y, feature_names, max_features=15
        )
        
        if features_mask is not None:
            if hasattr(X, 'iloc'):  # Se for DataFrame
                X_selected = X.iloc[:, features_mask]
            else:  # Se for array numpy
                X_selected = X[:, features_mask]
            print(f"   🎯 SHAP selecionou {X_selected.shape[1]} de {X.shape[1]} features")
            return X_selected, np.array(features_mask), None
    
    elif metodo == 'rfe':
        # Recursive Feature Elimination com validação cruzada
        rf_selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        selector = RFECV(estimator=rf_selector, step=1, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        X_selected = selector.fit_transform(X, y)
        
        features_selecionadas = selector.support_
        print(f"   🎯 RFE selecionou {X_selected.shape[1]} de {X.shape[1]} features")
        
        return X_selected, features_selecionadas, selector
    
    elif metodo == 'kbest':
        # Selecionar K melhores features baseado em F-score
        k_features = min(15, X.shape[1])  # Máximo 15 features
        selector = SelectKBest(score_func=f_regression, k=k_features)
        X_selected = selector.fit_transform(X, y)
        
        features_selecionadas = selector.get_support()
        print(f"   🎯 K-Best selecionou {X_selected.shape[1]} features")
        
        return X_selected, features_selecionadas, selector
    
    # Fallback: usar todas as features
    return X, np.ones(X.shape[1], dtype=bool), None

def criar_estratos_preco(y, n_estratos=5):
    """Cria estratos baseados em faixas de preço para validação estratificada"""
    try:
        # Usar percentis para criar faixas equilibradas
        percentis = np.linspace(0, 100, n_estratos + 1)
        limites = np.percentile(y, percentis)
        
        # Criar labels para cada estrato
        estratos = np.digitize(y, limites[1:-1])
        
        # Garantir que os valores estejam no range correto
        estratos = np.clip(estratos, 0, n_estratos - 1)
        
        return estratos
    except Exception as e:
        logging.warning(f"Erro ao criar estratos: {e}. Usando estratificação simples.")
        # Fallback: dividir em quartis simples
        q25, q50, q75 = np.percentile(y, [25, 50, 75])
        estratos = np.where(y <= q25, 0, 
                           np.where(y <= q50, 1,
                                   np.where(y <= q75, 2, 3)))
        return estratos

def validacao_cruzada_estratificada(X, y, modelo, n_splits=5):
    """Realiza validação cruzada estratificada por faixa de preço"""
    print(f"   📊 Realizando validação cruzada estratificada com {n_splits} splits...")
    
    # Criar estratos baseados em faixas de preço
    estratos = criar_estratos_preco(y, n_estratos=5)
    
    # Usar StratifiedKFold baseado nos estratos de preço
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    scores_mae = []
    scores_rmse = []
    scores_r2 = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, estratos)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Estatísticas dos estratos no fold
        estratos_train = estratos[train_idx]
        estratos_val = estratos[val_idx]
        
        print(f"      Fold {fold+1}: Train estratos {np.bincount(estratos_train)}, "
              f"Val estratos {np.bincount(estratos_val)}")
        
        # Treinar modelo no fold
        modelo_fold = modelo.__class__(**modelo.get_params())
        modelo_fold.fit(X_train_fold, y_train_fold)
        
        # Predizer no conjunto de validação
        y_pred_fold = modelo_fold.predict(X_val_fold)
        
        # Calcular métricas
        mae_fold = mean_absolute_error(y_val_fold, y_pred_fold)
        rmse_fold = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
        r2_fold = r2_score(y_val_fold, y_pred_fold)
        
        scores_mae.append(mae_fold)
        scores_rmse.append(rmse_fold)
        scores_r2.append(r2_fold)
        
        print(f"         Métricas: MAE={mae_fold:.2f}, RMSE={rmse_fold:.2f}, R²={r2_fold:.4f}")
    
    print(f"   📊 CV Estratificada - Médias: MAE={np.mean(scores_mae):.2f}±{np.std(scores_mae):.2f}")
    print(f"                                RMSE={np.mean(scores_rmse):.2f}±{np.std(scores_rmse):.2f}")
    print(f"                                R²={np.mean(scores_r2):.4f}±{np.std(scores_r2):.4f}")
    
    return {
        'mae_mean': np.mean(scores_mae), 'mae_std': np.std(scores_mae),
        'rmse_mean': np.mean(scores_rmse), 'rmse_std': np.std(scores_rmse),
        'r2_mean': np.mean(scores_r2), 'r2_std': np.std(scores_r2),
        'estratos_info': {
            'n_estratos': len(np.unique(estratos)),
            'distribuicao': np.bincount(estratos).tolist()
        }
    }

def validacao_cruzada_temporal(X, y, modelo, n_splits=5):
    """Realiza validação cruzada temporal respeitando ordem cronológica"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    scores_mae = []
    scores_rmse = []
    scores_r2 = []
    
    print(f"   ⏰ Realizando validação cruzada temporal com {n_splits} splits...")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Treinar modelo no fold
        modelo_fold = modelo.__class__(**modelo.get_params())
        modelo_fold.fit(X_train_fold, y_train_fold)
        
        # Predizer no conjunto de validação
        y_pred_fold = modelo_fold.predict(X_val_fold)
        
        # Calcular métricas
        mae_fold = mean_absolute_error(y_val_fold, y_pred_fold)
        rmse_fold = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
        r2_fold = r2_score(y_val_fold, y_pred_fold)
        
        scores_mae.append(mae_fold)
        scores_rmse.append(rmse_fold)
        scores_r2.append(r2_fold)
        
        print(f"      Fold {fold+1}: MAE={mae_fold:.2f}, RMSE={rmse_fold:.2f}, R²={r2_fold:.4f}")
    
    print(f"   📊 CV Temporal - Médias: MAE={np.mean(scores_mae):.2f}±{np.std(scores_mae):.2f}")
    print(f"                           RMSE={np.mean(scores_rmse):.2f}±{np.std(scores_rmse):.2f}")
    print(f"                           R²={np.mean(scores_r2):.4f}±{np.std(scores_r2):.4f}")
    
    return {
        'mae_mean': np.mean(scores_mae), 'mae_std': np.std(scores_mae),
        'rmse_mean': np.mean(scores_rmse), 'rmse_std': np.std(scores_rmse),
        'r2_mean': np.mean(scores_r2), 'r2_std': np.std(scores_r2)
    }

def otimizar_hiperparametros(X_train, y_train, modelo_tipo='random_forest'):
    """Otimiza hiperparâmetros usando Grid Search"""
    print(f"   🔧 Otimizando hiperparâmetros para {modelo_tipo}...")
    logging.info(f"Iniciando otimização de hiperparâmetros para {modelo_tipo}")
    
    if modelo_tipo == 'random_forest':
        modelo = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [20, 25, 30],
            'min_samples_split': [2, 3],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
    elif modelo_tipo == 'xgboost':
        modelo = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
    
    grid_search = GridSearchCV(
        estimator=modelo,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"   ✅ Melhores parâmetros encontrados:")
    for param, value in grid_search.best_params_.items():
        print(f"      {param}: {value}")
    
    logging.info(f"Melhores parâmetros: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

def analisar_residuos(y_true, y_pred, nome_modelo):
    """Analisa resíduos do modelo para diagnósticos"""
    residuos = y_true - y_pred
    
    # Estatísticas dos resíduos
    mean_residuo = np.mean(residuos)
    std_residuo = np.std(residuos)
    
    # Teste de normalidade (Shapiro-Wilk para amostra pequena)
    if len(residuos) < 5000:
        shapiro_stat, shapiro_p = stats.shapiro(residuos[:5000])  # Limitar amostra
        normalidade = "Normal" if shapiro_p > 0.05 else "Não Normal"
    else:
        normalidade = "Amostra muito grande para teste"
    
    # Heterocedasticidade (teste simples)
    residuos_abs = np.abs(residuos)
    correlation_het = np.corrcoef(y_pred, residuos_abs)[0, 1]
    heterocedasticidade = "Presente" if abs(correlation_het) > 0.1 else "Ausente"
    
    print(f"   📊 Análise de Resíduos - {nome_modelo}:")
    print(f"      Média dos resíduos: {mean_residuo:.4f}")
    print(f"      Desvio padrão: {std_residuo:.2f}")
    print(f"      Normalidade: {normalidade}")
    print(f"      Heterocedasticidade: {heterocedasticidade}")
    
    return {
        'mean_residuo': mean_residuo,
        'std_residuo': std_residuo,
        'normalidade': normalidade,
        'heterocedasticidade': heterocedasticidade
    }

def calcular_intervalos_predicao(modelo, X_sample, dados_categoria=None, nivel_confianca=0.95):
    """Calcula intervalos de predição mais robustos"""
    if hasattr(modelo, 'estimators_'):  # Para Random Forest
        # Usar predições de todas as árvores
        tree_predictions = np.array([tree.predict(X_sample) for tree in modelo.estimators_])
        
        # Calcular percentis para intervalo de confiança
        alpha = 1 - nivel_confianca
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        prediction_mean = np.mean(tree_predictions, axis=0)
        prediction_lower = np.percentile(tree_predictions, lower_percentile, axis=0)
        prediction_upper = np.percentile(tree_predictions, upper_percentile, axis=0)
        
        return prediction_mean, prediction_lower, prediction_upper
    
    else:
        # Para outros modelos, usar estimativa baseada na variabilidade do mercado
        prediction = modelo.predict(X_sample)
        
        if dados_categoria is not None:
            # Usar variabilidade real dos dados do mercado
            market_std = dados_categoria['price'].std()
            market_mean = dados_categoria['price'].mean()
            
            # Calcular margem de erro baseada na volatilidade do mercado
            relative_std = market_std / market_mean
            prediction_std = prediction * relative_std
            
            # Aplicar fator z para nível de confiança
            z_score = 1.96 if nivel_confianca == 0.95 else 2.58  # 95% ou 99%
            margin = prediction_std * z_score
        else:
            # Fallback: usar 10% da predição como margem
            margin = prediction * 0.1
        
        return prediction, prediction - margin, prediction + margin

# ============================================================================
# 4. MONITORAMENTO E DRIFT DETECTION
# ============================================================================

def calcular_drift_estatistico(dados_referencia, dados_atuais, threshold=0.05):
    """Calcula drift usando testes estatísticos"""
    drift_results = {}
    
    for coluna in dados_referencia.columns:
        if dados_referencia[coluna].dtype in ['int64', 'float64']:
            try:
                # Teste Kolmogorov-Smirnov para variáveis numéricas
                statistic, p_value = stats.ks_2samp(
                    dados_referencia[coluna].dropna(), 
                    dados_atuais[coluna].dropna()
                )
                
                drift_results[coluna] = {
                    'test': 'ks_test',
                    'statistic': statistic,
                    'p_value': p_value,
                    'drift_detected': p_value < threshold,
                    'severity': 'high' if p_value < 0.001 else 'medium' if p_value < 0.01 else 'low'
                }
            except Exception as e:
                drift_results[coluna] = {
                    'test': 'ks_test',
                    'error': str(e),
                    'drift_detected': False
                }
    
    return drift_results

def monitorar_drift_dados(dados_referencia, dados_atuais, features_importantes=None):
    """Monitora drift nos dados usando múltiplas abordagens"""
    print_section("MONITORAMENTO DE DRIFT DOS DADOS")
    
    # 1. Drift estatístico básico
    drift_stats = calcular_drift_estatistico(dados_referencia, dados_atuais)
    
    # 2. Drift usando Evidently (se disponível)
    drift_evidently = None
    if EVIDENTLY_AVAILABLE:
        try:
            # Preparar dados para Evidently
            reference_data = dados_referencia.select_dtypes(include=[np.number])
            current_data = dados_atuais.select_dtypes(include=[np.number])
            
            # Garantir que ambos datasets tenham as mesmas colunas
            common_columns = reference_data.columns.intersection(current_data.columns)
            reference_data = reference_data[common_columns]
            current_data = current_data[common_columns]
            
            # Criar relatório de drift
            drift_report = Report(metrics=[
                DatasetDriftMetric(),
                DatasetMissingValuesMetric()
            ])
            
            drift_report.run(
                reference_data=reference_data,
                current_data=current_data
            )
            
            # Salvar relatório
            drift_report.save_html("drift_report.html")
            print(f"   📊 Relatório Evidently salvo em: drift_report.html")
            
            drift_evidently = "Relatório gerado com sucesso"
            
        except Exception as e:
            print(f"   ⚠️ Erro no Evidently: {e}")
            drift_evidently = f"Erro: {e}"
    
    # 3. Analisar resultados
    print(f"\n   📊 RESULTADOS DO MONITORAMENTO:")
    
    drifts_detectados = 0
    features_com_drift = []
    
    for feature, result in drift_stats.items():
        if result.get('drift_detected', False):
            drifts_detectados += 1
            features_com_drift.append(feature)
            severity = result.get('severity', 'unknown')
            p_value = result.get('p_value', 0)
            
            icon = "🔴" if severity == 'high' else "🟡" if severity == 'medium' else "🟢"
            print(f"   {icon} {feature}: p-value={p_value:.6f} ({severity})")
    
    if drifts_detectados == 0:
        print(f"   ✅ Nenhum drift significativo detectado")
    else:
        print(f"   ⚠️ {drifts_detectados} features com drift detectado")
        
        if features_importantes:
            drift_importantes = [f for f in features_com_drift if f in features_importantes]
            if drift_importantes:
                print(f"   🚨 Features importantes com drift: {drift_importantes}")
    
    # 4. Calcular score geral de drift
    total_features = len(drift_stats)
    drift_score = (drifts_detectados / total_features) * 100 if total_features > 0 else 0
    
    print(f"\n   📈 SCORE GERAL DE DRIFT: {drift_score:.1f}%")
    
    if drift_score > 30:
        print(f"   🚨 ALERTA: Alto nível de drift detectado - considere retreinar o modelo")
    elif drift_score > 15:
        print(f"   ⚠️ ATENÇÃO: Drift moderado detectado - monitorar de perto")
    else:
        print(f"   ✅ Nível de drift aceitável")
    
    return {
        'drift_score': drift_score,
        'features_com_drift': features_com_drift,
        'total_features': total_features,
        'drift_details': drift_stats,
        'evidently_status': drift_evidently,
        'recomendacao': 'retreinar' if drift_score > 30 else 'monitorar' if drift_score > 15 else 'ok'
    }

def salvar_dados_referencia(dados, caminho='dados_referencia.pkl'):
    """Salva dados de referência para monitoramento futuro"""
    try:
        with open(caminho, 'wb') as f:
            pickle.dump(dados, f)
        print(f"   💾 Dados de referência salvos em: {caminho}")
        return True
    except Exception as e:
        print(f"   ❌ Erro ao salvar dados de referência: {e}")
        return False

def carregar_dados_referencia(caminho='dados_referencia.pkl'):
    """Carrega dados de referência para comparação"""
    try:
        with open(caminho, 'rb') as f:
            dados = pickle.load(f)
        print(f"   📂 Dados de referência carregados de: {caminho}")
        return dados
    except Exception as e:
        print(f"   ⚠️ Não foi possível carregar dados de referência: {e}")
        return None

# ============================================================================
# 5. SISTEMA DE RETREINAMENTO AUTOMÁTICO
# ============================================================================

class ModeloRetreinamento:
    """Classe para gerenciar retreinamento automático do modelo"""
    
    def __init__(self, caminho_modelo='modelo_pricing.pkl', caminho_config='config_retreinamento.json'):
        self.caminho_modelo = caminho_modelo
        self.caminho_config = caminho_config
        self.config = self.carregar_config()
    
    def carregar_config(self):
        """Carrega configurações de retreinamento"""
        config_default = {
            'threshold_drift': 20,  # % de drift para triggerar retreinamento
            'min_dias_entre_treinos': 7,  # Mínimo de dias entre retreinamentos
            'min_novos_dados': 1000,  # Mínimo de novos registros
            'backup_modelos': True,
            'notificar_retreino': True
        }
        
        try:
            with open(self.caminho_config, 'r') as f:
                config_usuario = json.load(f)
            config_default.update(config_usuario)
        except FileNotFoundError:
            # Criar arquivo de config padrão
            with open(self.caminho_config, 'w') as f:
                json.dump(config_default, f, indent=2)
            print(f"   📝 Arquivo de configuração criado: {self.caminho_config}")
        
        return config_default
    
    def verificar_necessidade_retreino(self, drift_score, novos_dados, ultimo_treino=None):
        """Verifica se é necessário retreinar o modelo"""
        criterios = {
            'drift_alto': drift_score > self.config['threshold_drift'],
            'dados_suficientes': len(novos_dados) >= self.config['min_novos_dados'],
            'tempo_adequado': True  # Por padrão, assume que passou tempo suficiente
        }
        
        if ultimo_treino:
            dias_desde_ultimo = (datetime.now() - ultimo_treino).days
            criterios['tempo_adequado'] = dias_desde_ultimo >= self.config['min_dias_entre_treinos']
        
        necessita_retreino = all(criterios.values())
        
        print(f"   🔍 CRITÉRIOS PARA RETREINAMENTO:")
        for criterio, valor in criterios.items():
            icon = "✅" if valor else "❌"
            print(f"      {icon} {criterio}: {valor}")
        
        return necessita_retreino, criterios
    
    def salvar_modelo(self, modelo, scaler, features, metadata=None):
        """Salva modelo treinado com metadata"""
        modelo_data = {
            'modelo': modelo,
            'scaler': scaler,
            'features': features,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        
        # Backup do modelo anterior se configurado
        if self.config['backup_modelos'] and os.path.exists(self.caminho_modelo):
            backup_path = f"{self.caminho_modelo}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(self.caminho_modelo, backup_path)
            print(f"   📦 Backup do modelo anterior: {backup_path}")
        
        # Salvar novo modelo
        with open(self.caminho_modelo, 'wb') as f:
            pickle.dump(modelo_data, f)
        
        print(f"   💾 Modelo salvo: {self.caminho_modelo}")
        return True
    
    def carregar_modelo(self):
        """Carrega modelo salvo"""
        try:
            with open(self.caminho_modelo, 'rb') as f:
                modelo_data = pickle.load(f)
            
            print(f"   📂 Modelo carregado de: {self.caminho_modelo}")
            print(f"   📅 Treinado em: {modelo_data['timestamp']}")
            
            return modelo_data
        except Exception as e:
            print(f"   ❌ Erro ao carregar modelo: {e}")
            return None

def agendar_retreinamento_automatico():
    """Agenda verificações periódicas para retreinamento"""
    def verificar_e_retreinar():
        print_section("VERIFICAÇÃO AUTOMÁTICA DE RETREINAMENTO")
        print(f"   🕐 Executado em: {datetime.now()}")
        
        # Aqui você implementaria a lógica para:
        # 1. Carregar novos dados
        # 2. Verificar drift
        # 3. Retreinar se necessário
        
        print("   ✅ Verificação concluída")
    
    # Agendar para executar diariamente às 2:00 AM
    schedule.every().day.at("02:00").do(verificar_e_retreinar)
    
    print("   ⏰ Retreinamento automático agendado para 02:00 diariamente")
    print("   💡 Para ativar, execute: python -c 'import schedule; import time; while True: schedule.run_pending(); time.sleep(60)'")

def treinar_modelos_multiplos(df):
    """Treina múltiplos modelos e compara performance"""
    print_section("TREINAMENTO DE MÚLTIPLOS MODELOS COM OTIMIZAÇÃO")
    logging.info("Iniciando treinamento de múltiplos modelos")
    
    # Selecionar features para o modelo
    feature_columns = [
        'freight_value', 'product_weight_g', 'product_length_cm', 
        'product_height_cm', 'product_width_cm', 'product_volume',
        'density', 'review_score', 'num_reviews', 'quality_score',
        'order_month', 'order_dayofweek', 'order_hour', 'is_weekend',
        'month_sin', 'month_cos', 'same_state',
        'category_encoded', 'customer_state_encoded', 'seller_state_encoded',
        'category_price_mean', 'category_price_std', 'category_count',
        'category_review_mean'
    ]
    
    # Verificar quais features existem
    features_disponiveis = [f for f in feature_columns if f in df.columns]
    print(f"   📊 Features disponíveis: {len(features_disponiveis)}")
    
    # Preparar dados
    X = df[features_disponiveis]
    y = df['price']
    
    # Remover valores infinitos e NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   🎯 Treino: {X_train.shape[0]} amostras")
    print(f"   🎯 Teste: {X_test.shape[0]} amostras")
    
    # Seleção automática de features
    print(f"\n   🎯 SELEÇÃO AUTOMÁTICA DE FEATURES:")
    metodo_selecao = 'shap' if SHAP_AVAILABLE else 'rfe'
    X_train_selected, features_mask, feature_selector = selecionar_features_automatico(
        X_train, y_train, metodo=metodo_selecao
    )
    if feature_selector is not None and hasattr(feature_selector, 'transform'):
        X_test_selected = feature_selector.transform(X_test)
    elif metodo_selecao == 'shap' and features_mask is not None:
        if hasattr(X_test, 'iloc'):  # Se for DataFrame
            X_test_selected = X_test.iloc[:, features_mask]
        else:  # Se for array numpy
            X_test_selected = X_test[:, features_mask]
    else:
        X_test_selected = X_test
    
    features_selecionadas = [features_disponiveis[i] for i, selected in enumerate(features_mask) if selected]
    print(f"   ✅ Features selecionadas: {features_selecionadas}")
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Dicionário para armazenar resultados dos modelos
    resultados_modelos = {}
    
    # 1. RANDOM FOREST OTIMIZADO
    print(f"\n   🌲 TREINANDO RANDOM FOREST:")
    try:
        rf_otimizado = otimizar_hiperparametros(X_train_scaled, y_train, 'random_forest')
        rf_otimizado.fit(X_train_scaled, y_train)
        
        # Predições
        y_train_pred_rf = rf_otimizado.predict(X_train_scaled)
        y_test_pred_rf = rf_otimizado.predict(X_test_scaled)
        
        # Métricas
        mae_test_rf = mean_absolute_error(y_test, y_test_pred_rf)
        rmse_test_rf = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
        r2_test_rf = r2_score(y_test, y_test_pred_rf)
        mape_test_rf = np.mean(np.abs((y_test - y_test_pred_rf) / y_test)) * 100
        
        # Análise de resíduos
        residuos_rf = analisar_residuos(y_test, y_test_pred_rf, "Random Forest")
        
        # Validação cruzada temporal e estratificada
        cv_results_temporal = validacao_cruzada_temporal(X_train_scaled, y_train, rf_otimizado)
        cv_results_estratificada = validacao_cruzada_estratificada(X_train_scaled, y_train, rf_otimizado)
        
        # Combinar resultados (usar média das duas validações)
        cv_results_rf = {
            'mae_mean': (cv_results_temporal['mae_mean'] + cv_results_estratificada['mae_mean']) / 2,
            'rmse_mean': (cv_results_temporal['rmse_mean'] + cv_results_estratificada['rmse_mean']) / 2,
            'r2_mean': (cv_results_temporal['r2_mean'] + cv_results_estratificada['r2_mean']) / 2,
            'mae_std': cv_results_estratificada['mae_std'],
            'rmse_std': cv_results_estratificada['rmse_std'],
            'r2_std': cv_results_estratificada['r2_std'],
            'temporal': cv_results_temporal,
            'estratificada': cv_results_estratificada
        }
        
        resultados_modelos['Random Forest'] = {
            'modelo': rf_otimizado,
            'mae_test': mae_test_rf,
            'rmse_test': rmse_test_rf,
            'r2_test': r2_test_rf,
            'mape_test': mape_test_rf,
            'residuos': residuos_rf,
            'cv_results': cv_results_rf,
            'feature_importance': pd.DataFrame({
                'Feature': features_selecionadas,
                'Importance': rf_otimizado.feature_importances_
            }).sort_values('Importance', ascending=False)
        }
        
        print(f"      ✅ RF - MAE: R$ {mae_test_rf:.2f} | RMSE: R$ {rmse_test_rf:.2f} | R²: {r2_test_rf:.4f}")
        
    except Exception as e:
        print(f"      ❌ Erro no Random Forest: {e}")
        logging.error(f"Erro no Random Forest: {e}")
    
    # 2. XGBOOST OTIMIZADO
    print(f"\n   🚀 TREINANDO XGBOOST:")
    try:
        xgb_otimizado = otimizar_hiperparametros(X_train_scaled, y_train, 'xgboost')
        xgb_otimizado.fit(X_train_scaled, y_train)
        
        # Predições
        y_train_pred_xgb = xgb_otimizado.predict(X_train_scaled)
        y_test_pred_xgb = xgb_otimizado.predict(X_test_scaled)
        
        # Métricas
        mae_test_xgb = mean_absolute_error(y_test, y_test_pred_xgb)
        rmse_test_xgb = np.sqrt(mean_squared_error(y_test, y_test_pred_xgb))
        r2_test_xgb = r2_score(y_test, y_test_pred_xgb)
        mape_test_xgb = np.mean(np.abs((y_test - y_test_pred_xgb) / y_test)) * 100
        
        # Análise de resíduos
        residuos_xgb = analisar_residuos(y_test, y_test_pred_xgb, "XGBoost")
        
        # Validação cruzada temporal e estratificada
        cv_results_temporal = validacao_cruzada_temporal(X_train_scaled, y_train, xgb_otimizado)
        cv_results_estratificada = validacao_cruzada_estratificada(X_train_scaled, y_train, xgb_otimizado)
        
        # Combinar resultados (usar média das duas validações)
        cv_results_xgb = {
            'mae_mean': (cv_results_temporal['mae_mean'] + cv_results_estratificada['mae_mean']) / 2,
            'rmse_mean': (cv_results_temporal['rmse_mean'] + cv_results_estratificada['rmse_mean']) / 2,
            'r2_mean': (cv_results_temporal['r2_mean'] + cv_results_estratificada['r2_mean']) / 2,
            'mae_std': cv_results_estratificada['mae_std'],
            'rmse_std': cv_results_estratificada['rmse_std'],
            'r2_std': cv_results_estratificada['r2_std'],
            'temporal': cv_results_temporal,
            'estratificada': cv_results_estratificada
        }
        
        # Feature importance para XGBoost
        if hasattr(xgb_otimizado, 'feature_importances_'):
            feature_importance_xgb = pd.DataFrame({
                'Feature': features_selecionadas,
                'Importance': xgb_otimizado.feature_importances_
            }).sort_values('Importance', ascending=False)
        else:
            feature_importance_xgb = pd.DataFrame()
        
        resultados_modelos['XGBoost'] = {
            'modelo': xgb_otimizado,
            'mae_test': mae_test_xgb,
            'rmse_test': rmse_test_xgb,
            'r2_test': r2_test_xgb,
            'mape_test': mape_test_xgb,
            'residuos': residuos_xgb,
            'cv_results': cv_results_xgb,
            'feature_importance': feature_importance_xgb
        }
        
        print(f"      ✅ XGB - MAE: R$ {mae_test_xgb:.2f} | RMSE: R$ {rmse_test_xgb:.2f} | R²: {r2_test_xgb:.4f}")
        
    except Exception as e:
        print(f"      ❌ Erro no XGBoost: {e}")
        logging.error(f"Erro no XGBoost: {e}")
    
    # 3. COMPARAÇÃO DE MODELOS
    print(f"\n   📊 COMPARAÇÃO DE MODELOS:")
    print("   " + "="*70)
    print("   MODELO          | MAE       | RMSE      | R²      | MAPE     | CV R²")
    print("   " + "="*70)
    
    melhor_modelo = None
    melhor_r2 = -np.inf
    
    for nome, resultado in resultados_modelos.items():
        cv_r2 = resultado['cv_results']['r2_mean']
        print(f"   {nome:<15} | R$ {resultado['mae_test']:6.2f} | R$ {resultado['rmse_test']:6.2f} | "
              f"{resultado['r2_test']:6.4f} | {resultado['mape_test']:6.2f}% | {cv_r2:6.4f}")
        
        # Selecionar melhor modelo baseado no R² de validação cruzada
        if cv_r2 > melhor_r2:
            melhor_r2 = cv_r2
            melhor_modelo = nome
    
    print("   " + "="*70)
    print(f"   🏆 MELHOR MODELO: {melhor_modelo} (CV R²: {melhor_r2:.4f})")
    
    # Retornar o melhor modelo e informações
    modelo_final = resultados_modelos[melhor_modelo]['modelo']
    metricas_final = {
        'mae_test': resultados_modelos[melhor_modelo]['mae_test'],
        'rmse_test': resultados_modelos[melhor_modelo]['rmse_test'],
        'r2_test': resultados_modelos[melhor_modelo]['r2_test'],
        'mape_test': resultados_modelos[melhor_modelo]['mape_test']
    }
    
    feature_importance_final = resultados_modelos[melhor_modelo]['feature_importance']
    
    print(f"\n   🎯 TOP 5 FEATURES MAIS IMPORTANTES ({melhor_modelo}):")
    if not feature_importance_final.empty:
        for i, row in feature_importance_final.head(5).iterrows():
            print(f"      {row['Feature']}: {row['Importance']:.4f}")
    
    logging.info(f"Melhor modelo selecionado: {melhor_modelo}")
    logging.info(f"Métricas finais: {metricas_final}")
    
    return (modelo_final, scaler, features_selecionadas, feature_importance_final, 
            metricas_final, resultados_modelos, feature_selector)

# Manter função original para compatibilidade
def treinar_modelo(df):
    """Função wrapper para manter compatibilidade"""
    resultado = treinar_modelos_multiplos(df)
    return resultado[:5]  # Retornar apenas os 5 primeiros elementos

# ============================================================================
# 4. SISTEMA DE RECOMENDAÇÃO DE PREÇOS
# ============================================================================

def calcular_confianca_predicao(modelo, X_sample_scaled, dados_categoria):
    """Calcula confiança baseada na variabilidade das árvores e qualidade dos dados"""
    # Usar intervalos de predição robustos
    if hasattr(modelo, 'estimators_'):  # Random Forest
        tree_predictions = [tree.predict(X_sample_scaled.reshape(1, -1))[0] 
                           for tree in modelo.estimators_]
        
        mean_pred = np.mean(tree_predictions)
        std_pred = np.std(tree_predictions)
        
        # Calcular coeficiente de variação
        cv = std_pred / abs(mean_pred) if mean_pred != 0 else 1
        
        # Confiança base
        confianca_base = max(70, min(95, 100 * (1 - cv * 0.3)))
        
        return_predictions = tree_predictions
    
    else:  # XGBoost ou outros modelos
        # Para modelos sem ensemble, simular múltiplas predições com bootstrap
        prediction = modelo.predict(X_sample_scaled.reshape(1, -1))[0]
        
        # Gerar estimativas de variabilidade baseadas nos dados de categoria
        variabilidade_mercado = dados_categoria['price'].std()
        media_mercado = dados_categoria['price'].mean()
        
        # Simular 100 predições com ruído baseado na variabilidade do mercado
        np.random.seed(42)  # Para reprodutibilidade
        noise_factor = min(0.1, variabilidade_mercado / media_mercado)  # Máximo 10% de ruído
        simulated_predictions = []
        
        for _ in range(100):
            noise = np.random.normal(0, prediction * noise_factor)
            simulated_predictions.append(prediction + noise)
        
        # Confiança baseada na estabilidade do mercado
        variabilidade_relativa = variabilidade_mercado / media_mercado
        confianca_base = max(75, min(95, 100 * (1 - variabilidade_relativa * 0.5)))
        
        return_predictions = simulated_predictions
    
    # Ajustes baseados na qualidade dos dados
    num_registros = len(dados_categoria)
    num_vendedores = dados_categoria['seller_id'].nunique() if 'seller_id' in dados_categoria.columns else 1
    variabilidade_precos = dados_categoria['price'].std() / dados_categoria['price'].mean()
    
    # Bônus por quantidade de dados
    if num_registros >= 100:
        confianca_base += 5
    elif num_registros >= 50:
        confianca_base += 3
    elif num_registros < 20:
        confianca_base -= 5
    
    # Bônus por diversidade de vendedores
    if num_vendedores >= 10:
        confianca_base += 3
    elif num_vendedores >= 5:
        confianca_base += 2
    
    # Penalidade por alta variabilidade
    if variabilidade_precos > 1.0:
        confianca_base -= 3
    
    # Garantir range correto
    confianca_final = max(75, min(98, confianca_base))
    
    return confianca_final, return_predictions

def validar_e_ajustar_preco(preco_predito, dados_categoria):
    """Valida e ajusta o preço predito"""
    precos_mercado = dados_categoria['price']
    preco_min = precos_mercado.min()
    preco_max = precos_mercado.max()
    preco_mediano = precos_mercado.median()
    
    # Limites razoáveis
    limite_inferior = preco_min * 0.6
    limite_superior = preco_max * 1.8
    
    preco_ajustado = preco_predito
    ajuste_aplicado = ""
    
    if preco_predito < limite_inferior:
        preco_ajustado = preco_min * 0.85
        ajuste_aplicado = f"Ajustado para cima: era R$ {preco_predito:.2f}, muito baixo"
    elif preco_predito > limite_superior:
        preco_ajustado = preco_max * 1.15
        ajuste_aplicado = f"Ajustado para baixo: era R$ {preco_predito:.2f}, muito alto"
    
    return preco_ajustado, ajuste_aplicado

def gerar_justificativas_especificas(categoria_nome, preco_recomendado, dados_categoria):
    """Gera justificativas específicas e únicas para cada categoria"""
    justificativas = []
    
    # Análise dos preços
    precos = dados_categoria['price']
    preco_min = precos.min()
    preco_max = precos.max()
    preco_medio = precos.mean()
    preco_mediano = precos.median()
    
    # 1. POSICIONAMENTO ESPECÍFICO
    if preco_recomendado < preco_min:
        diferenca = ((preco_min - preco_recomendado) / preco_min * 100)
        justificativas.append(f"💰 Estratégia agressiva: {diferenca:.0f}% abaixo do menor preço atual")
    elif preco_recomendado > preco_max:
        diferenca = ((preco_recomendado - preco_max) / preco_max * 100)
        justificativas.append(f"💎 Posicionamento premium: {diferenca:.0f}% acima do maior preço")
    elif preco_recomendado <= preco_mediano:
        diferenca = ((preco_mediano - preco_recomendado) / preco_mediano * 100)
        justificativas.append(f"⚡ Competitivo: {diferenca:.0f}% abaixo da mediana")
    else:
        diferenca = ((preco_recomendado - preco_medio) / preco_medio * 100)
        justificativas.append(f"📈 Margem otimizada: {diferenca:.0f}% acima da média")
    
    # 2. ANÁLISE ESPECÍFICA POR CATEGORIA
    categoria_lower = categoria_nome.lower()
    
    if 'informatica' in categoria_lower or 'telefonia' in categoria_lower:
        variabilidade = (preco_max - preco_min) / preco_medio * 100
        if variabilidade > 100:
            justificativas.append(f"🔧 Tech: alta variação ({variabilidade:.0f}%) permite diferenciação")
        else:
            justificativas.append("🔧 Tech: mercado padronizado, preço competitivo necessário")
    
    elif 'casa' in categoria_lower or 'moveis' in categoria_lower:
        peso_medio = dados_categoria['product_weight_g'].mean() if 'product_weight_g' in dados_categoria.columns else 0
        if peso_medio > 2000:
            justificativas.append(f"🏠 Móveis: produto pesado ({peso_medio/1000:.1f}kg), frete alto")
        else:
            justificativas.append("🏠 Casa: item decorativo, margem flexível")
    
    elif 'beleza' in categoria_lower or 'saude' in categoria_lower:
        nota_media = dados_categoria['review_score'].mean() if 'review_score' in dados_categoria.columns else 3.5
        if nota_media >= 4.2:
            justificativas.append(f"💄 Beleza: excelente reputação ({nota_media:.1f}★)")
        else:
            justificativas.append(f"💄 Beleza: qualidade moderada ({nota_media:.1f}★)")
    
    elif 'esporte' in categoria_lower:
        justificativas.append("🏃 Esporte: mercado competitivo, preço é fator decisivo")
    
    elif 'livros' in categoria_lower:
        justificativas.append("📚 Livros: margens baixas, volume é importante")
    
    elif 'auto' in categoria_lower:
        justificativas.append("🚗 Auto: compra técnica, cliente pesquisa preços")
    
    else:
        num_vendedores = dados_categoria['seller_id'].nunique() if 'seller_id' in dados_categoria.columns else 1
        if num_vendedores >= 15:
            justificativas.append(f"🏪 Categoria disputada: {num_vendedores} vendedores")
        else:
            justificativas.append(f"🏪 Categoria concentrada: {num_vendedores} vendedores")
    
    # 3. ANÁLISE DE QUALIDADE
    if 'review_score' in dados_categoria.columns:
        nota_media = dados_categoria['review_score'].mean()
        num_reviews = dados_categoria['num_reviews'].mean() if 'num_reviews' in dados_categoria.columns else 0
        
        if nota_media >= 4.5:
            justificativas.append(f"⭐ Qualidade excepcional: {nota_media:.1f}★ de avaliação")
        elif nota_media >= 4.0:
            justificativas.append(f"⭐ Boa qualidade: {nota_media:.1f}★ suporta o preço")
        elif nota_media < 3.5:
            justificativas.append(f"⚠️ Qualidade questionável: {nota_media:.1f}★ exige desconto")
    
    # 4. ANÁLISE DE MERCADO
    variabilidade_total = (preco_max - preco_min) / preco_medio * 100
    if variabilidade_total > 150:
        justificativas.append(f"📊 Mercado volátil: variação de {variabilidade_total:.0f}%")
    elif variabilidade_total < 50:
        justificativas.append("📊 Mercado estável: preços padronizados")
    
    return justificativas[:4]

# ============================================================================
# 5. ANÁLISE DE PRODUTOS
# ============================================================================

def analisar_produtos(df, modelo, scaler, features_disponiveis, feature_importance):
    """Analisa cada produto e gera recomendações"""
    print_section("ANÁLISE DE PRODUTOS E RECOMENDAÇÕES")
    
    # Agrupar por categoria
    produtos_por_categoria = df.groupby('product_category_name').agg({
        'price': ['count', 'mean', 'min', 'max', 'std'],
        'review_score': 'mean',
        'product_id': 'nunique'
    }).round(2)
    
    produtos_por_categoria.columns = ['count', 'price_mean', 'price_min', 'price_max', 
                                    'price_std', 'review_mean', 'unique_products']
    produtos_por_categoria = produtos_por_categoria.reset_index()
    
    # Filtrar categorias com dados suficientes (reduzindo limite para incluir mais categorias)
    produtos_por_categoria = produtos_por_categoria[produtos_por_categoria['count'] >= 5]
    produtos_por_categoria = produtos_por_categoria.sort_values('count', ascending=False)
    
    print(f"   📊 Analisando {len(produtos_por_categoria)} categorias de produtos")
    
    resultados = []
    
    for idx, categoria_info in produtos_por_categoria.head(50).iterrows():
        categoria = categoria_info['product_category_name']
        
        print(f"\n{Colors.CYAN}🔍 ANALISANDO: {Colors.YELLOW}{categoria.upper()}{Colors.RESET}")
        print(f"{Colors.BLUE}{'─'*50}{Colors.RESET}")
        
        # Filtrar dados da categoria
        dados_categoria = df[df['product_category_name'] == categoria]
        
        print(f"   📦 Registros: {len(dados_categoria)}")
        print(f"   🏪 Vendedores: {dados_categoria['seller_id'].nunique() if 'seller_id' in dados_categoria.columns else 'N/A'}")
        
        # Estatísticas dos preços
        precos_atuais = dados_categoria['price']
        print(f"   💰 Preços atuais:")
        print(f"      Mín: R$ {precos_atuais.min():.2f}")
        print(f"      Máx: R$ {precos_atuais.max():.2f}")
        print(f"      Médio: R$ {precos_atuais.mean():.2f}")
        print(f"      Mediano: R$ {precos_atuais.median():.2f}")
        
        # Preparar dados para predição
        dados_medios = dados_categoria[features_disponiveis].mean()
        X_pred = dados_medios.values.reshape(1, -1)
        X_pred = np.nan_to_num(X_pred, nan=0.0)
        X_pred_scaled = scaler.transform(X_pred)
        
        # Fazer predição
        preco_recomendado = modelo.predict(X_pred_scaled)[0]
        
        # Validar e ajustar preço
        preco_final, ajuste_msg = validar_e_ajustar_preco(preco_recomendado, dados_categoria)
        
        if ajuste_msg:
            print(f"   ⚠️ {ajuste_msg}")
            preco_recomendado = preco_final
        
        # Calcular confiança
        confianca, tree_preds = calcular_confianca_predicao(modelo, X_pred_scaled, dados_categoria)
        
        # Calcular variabilidade de forma consistente
        variabilidade_predicao = np.std(tree_preds) if len(tree_preds) > 1 else 0.0
        
        # Gerar justificativas específicas
        justificativas = gerar_justificativas_especificas(categoria, preco_recomendado, dados_categoria)
        
        # Exibir resultados
        print(f"\n   🎯 RECOMENDAÇÃO:")
        print(f"      💵 PREÇO RECOMENDADO: R$ {preco_recomendado:.2f}")
        print(f"      🎯 CONFIANÇA DA PREDIÇÃO: {confianca:.1f}%")
        print(f"      📊 Variação das predições: R$ {variabilidade_predicao:.2f}")
        
        print(f"\n   📋 JUSTIFICATIVAS:")
        for i, justificativa in enumerate(justificativas, 1):
            print(f"      {i}. {justificativa}")
        
        # Comparação com mercado
        diferenca_min = preco_recomendado - precos_atuais.min()
        diferenca_max = preco_recomendado - precos_atuais.max()
        diferenca_media = preco_recomendado - precos_atuais.mean()
        
        print(f"\n   📈 COMPARAÇÃO COM MERCADO:")
        print(f"      vs Menor preço: {diferenca_min:+.2f} R$")
        print(f"      vs Maior preço: {diferenca_max:+.2f} R$")
        print(f"      vs Preço médio: {diferenca_media:+.2f} R$")
        
        # Armazenar resultado
        resultado = {
            'Produto': categoria,
            'Preço_Recomendado': preco_recomendado,
            'Confiança_Predição': confianca,
            'Preço_Min_Mercado': precos_atuais.min(),
            'Preço_Max_Mercado': precos_atuais.max(),
            'Preço_Médio_Mercado': precos_atuais.mean(),
            'Preço_Mediano_Mercado': precos_atuais.median(),
            'Diferença_vs_Média': diferenca_media,
            'Variabilidade_Árvores': variabilidade_predicao,
            'Justificativas': justificativas,
            'Registros_Mercado': len(dados_categoria),
            'Vendedores_Únicos': dados_categoria['seller_id'].nunique() if 'seller_id' in dados_categoria.columns else 1
        }
        
        resultados.append(resultado)
    
    return resultados

# ============================================================================
# 6. VISUALIZAÇÕES
# ============================================================================

def criar_visualizacoes(resultados, metricas_modelo):
    """Cria visualizações interativas dos resultados com análise detalhada"""
    print_section("CRIANDO VISUALIZAÇÕES INTERATIVAS")
    
    if not resultados:
        print("   ❌ Nenhum resultado para visualizar")
        return
    
    df_resultados = pd.DataFrame(resultados)
    
    # Preparar dados para visualização detalhada
    produtos_detalhados = []
    for _, row in df_resultados.iterrows():
        justificativas_texto = "<br>".join([f"• {j}" for j in row['Justificativas']])
        
        produto_info = {
            'Produto': row['Produto'],
            'Preço_Anterior': row['Preço_Médio_Mercado'],
            'Preço_ML': row['Preço_Recomendado'],
            'Mudança_Percentual': ((row['Preço_Recomendado'] - row['Preço_Médio_Mercado']) / row['Preço_Médio_Mercado'] * 100),
            'Confiança': row['Confiança_Predição'],
            'Fatores_Considerados': justificativas_texto,
            'Variabilidade': row['Variabilidade_Árvores'],
            'Faixa_Min': row['Preço_Min_Mercado'],
            'Faixa_Max': row['Preço_Max_Mercado'],
            'Registros': row['Registros_Mercado'],
            'Vendedores': row['Vendedores_Únicos']
        }
        produtos_detalhados.append(produto_info)
    
    df_detalhado = pd.DataFrame(produtos_detalhados)
    
    # 1. GRÁFICO PRINCIPAL: Comparação Antes vs Depois com Detalhes
    fig_principal = go.Figure()
    
    # Barras dos preços anteriores (mercado)
    fig_principal.add_trace(go.Bar(
        name='Preço Médio Atual (Mercado)',
        x=df_detalhado['Produto'],
        y=df_detalhado['Preço_Anterior'],
        marker_color='lightblue',
        opacity=0.7,
        text=[f'R$ {p:.2f}' for p in df_detalhado['Preço_Anterior']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>' +
                     'Preço Médio Mercado: R$ %{y:.2f}<br>' +
                     '<extra></extra>'
    ))
    
    # Barras dos preços recomendados pelo ML
    fig_principal.add_trace(go.Bar(
        name='Preço Recomendado (ML)',
        x=df_detalhado['Produto'],
        y=df_detalhado['Preço_ML'],
        marker_color='darkgreen',
        opacity=0.9,
        text=[f'R$ {p:.2f}' for p in df_detalhado['Preço_ML']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>' +
                     'Preço ML: R$ %{y:.2f}<br>' +
                     'Confiança: %{customdata[0]:.1f}%<br>' +
                     'Mudança: %{customdata[1]:+.1f}%<br>' +
                     '<b>Fatores Considerados:</b><br>%{customdata[2]}<br>' +
                     '<extra></extra>',
        customdata=list(zip(df_detalhado['Confiança'], 
                           df_detalhado['Mudança_Percentual'],
                           df_detalhado['Fatores_Considerados']))
    ))
    
    # Linha mostrando faixa de preços do mercado
    fig_principal.add_trace(go.Scatter(
        name='Faixa Mín-Máx do Mercado',
        x=df_detalhado['Produto'],
        y=df_detalhado['Faixa_Min'],
        mode='markers',
        marker=dict(color='red', size=8, symbol='triangle-down'),
        showlegend=False,
        hovertemplate='Preço Mínimo: R$ %{y:.2f}<extra></extra>'
    ))
    
    fig_principal.add_trace(go.Scatter(
        name='Faixa Mín-Máx do Mercado',
        x=df_detalhado['Produto'],
        y=df_detalhado['Faixa_Max'],
        mode='markers',
        marker=dict(color='red', size=8, symbol='triangle-up'),
        hovertemplate='Preço Máximo: R$ %{y:.2f}<extra></extra>'
    ))
    
    fig_principal.update_layout(
        title={
            'text': 'Análise de Preços: Mercado Atual vs Recomendações do Machine Learning<br>' +
                   '<sub>Hover sobre as barras verdes para ver fatores considerados e taxa de confiança</sub>',
            'x': 0.5,
            'font': {'size': 16}
        },
        xaxis_title='Produtos',
        yaxis_title='Preço (R$)',
        height=700,
        width=1400,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        hovermode='closest'
    )
    
    fig_principal.update_xaxes(tickangle=45)
    
    # Salvar como arquivo HTML
    fig_principal.write_html("analise_precos_principal.html")
    print(f"   📊 Gráfico principal salvo em: analise_precos_principal.html")
    
    # 2. DASHBOARD DE ANÁLISE DETALHADA
    fig_dashboard = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            'Taxa de Confiança por Produto',
            'Mudança Percentual nos Preços',
            'Variabilidade das Predições',
            'Análise de Qualidade dos Dados',
            'Impacto Financeiro',
            'Performance do Modelo ML'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Criar nomes curtos para melhor visualização
    produtos_curtos = [p[:12] + '...' if len(p) > 12 else p for p in df_detalhado['Produto']]
    
    # 1. Taxa de Confiança
    colors_confianca = ['#2E8B57' if x >= 90 else '#FF8C00' if x >= 85 else '#DC143C' 
                       for x in df_detalhado['Confiança']]
    fig_dashboard.add_trace(
        go.Bar(name='Confiança', x=produtos_curtos, y=df_detalhado['Confiança'],
               marker_color=colors_confianca, opacity=0.8, showlegend=False,
               text=[f'{c:.1f}%' for c in df_detalhado['Confiança']],
               textposition='outside',
               hovertemplate='%{x}<br>Confiança: %{y:.1f}%<extra></extra>'),
        row=1, col=1
    )
    
    # 2. Mudança Percentual
    colors_mudanca = ['#2E8B57' if x > 0 else '#DC143C' if x < -5 else '#FF8C00' 
                     for x in df_detalhado['Mudança_Percentual']]
    fig_dashboard.add_trace(
        go.Bar(name='Mudança', x=produtos_curtos, y=df_detalhado['Mudança_Percentual'],
               marker_color=colors_mudanca, opacity=0.8, showlegend=False,
               text=[f'{m:+.1f}%' for m in df_detalhado['Mudança_Percentual']],
               textposition='outside',
               hovertemplate='%{x}<br>Mudança: %{y:+.1f}%<extra></extra>'),
        row=1, col=2
    )
    
    # 3. Variabilidade das Predições
    fig_dashboard.add_trace(
        go.Bar(name='Variabilidade', x=produtos_curtos, y=df_detalhado['Variabilidade'],
               marker_color='purple', opacity=0.7, showlegend=False,
               text=[f'R${v:.2f}' for v in df_detalhado['Variabilidade']],
               textposition='outside',
               hovertemplate='%{x}<br>Variabilidade: R$ %{y:.2f}<extra></extra>'),
        row=1, col=3
    )
    
    # 4. Qualidade dos Dados (Registros × Vendedores)
    fig_dashboard.add_trace(
        go.Scatter(x=df_detalhado['Registros'], y=df_detalhado['Vendedores'],
                  mode='markers+text', 
                  marker=dict(size=[r/20 for r in df_detalhado['Registros']], 
                            color=df_detalhado['Confiança'],
                            colorscale='RdYlGn', 
                            showscale=False,
                            line=dict(width=1, color='black')),
                  text=produtos_curtos,
                  textposition="middle center",
                  showlegend=False,
                  hovertemplate='%{text}<br>Registros: %{x}<br>Vendedores: %{y}<extra></extra>'),
        row=2, col=1
    )
    
    # 5. Impacto Financeiro
    impacto_financeiro = df_detalhado['Mudança_Percentual'] * df_detalhado['Preço_Anterior'] / 100
    colors_impacto = ['#2E8B57' if x > 0 else '#DC143C' for x in impacto_financeiro]
    fig_dashboard.add_trace(
        go.Bar(name='Impacto', x=produtos_curtos, y=impacto_financeiro,
               marker_color=colors_impacto, opacity=0.8, showlegend=False,
               text=[f'R${i:+.2f}' for i in impacto_financeiro],
               textposition='outside',
               hovertemplate='%{x}<br>Impacto: R$ %{y:+.2f}<extra></extra>'),
        row=2, col=2
    )
    
    # 6. Performance do Modelo
    metricas_nomes = ['MAE\n(R$)', 'RMSE\n(R$)', 'R²', 'MAPE\n(%)']
    metricas_valores = [
        metricas_modelo['mae_test'],
        metricas_modelo['rmse_test'],
        metricas_modelo['r2_test'],
        metricas_modelo['mape_test']
    ]
    
    fig_dashboard.add_trace(
        go.Bar(name='Métricas', x=metricas_nomes, y=metricas_valores,
               marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], 
               opacity=0.8, showlegend=False,
               text=[f'{m:.3f}' if m < 1 else f'{m:.1f}' for m in metricas_valores],
               textposition='outside',
               hovertemplate='%{x}: %{y}<extra></extra>'),
        row=2, col=3
    )
    
    # Atualizar layout do dashboard
    fig_dashboard.update_layout(
        height=900,
        width=1500,
        title_text="Dashboard Detalhado de Análise de Preços com Machine Learning",
        title_x=0.5,
        showlegend=False
    )
    
    # Configurar eixos
    fig_dashboard.update_xaxes(tickangle=45, row=1, col=1)
    fig_dashboard.update_xaxes(tickangle=45, row=1, col=2)
    fig_dashboard.update_xaxes(tickangle=45, row=1, col=3)
    fig_dashboard.update_xaxes(title_text="Nº de Registros", row=2, col=1)
    fig_dashboard.update_xaxes(tickangle=45, row=2, col=2)
    fig_dashboard.update_xaxes(tickangle=30, row=2, col=3)
    
    fig_dashboard.update_yaxes(title_text="Confiança (%)", row=1, col=1)
    fig_dashboard.update_yaxes(title_text="Mudança (%)", row=1, col=2)
    fig_dashboard.update_yaxes(title_text="Variabilidade (R$)", row=1, col=3)
    fig_dashboard.update_yaxes(title_text="Nº de Vendedores", row=2, col=1)
    fig_dashboard.update_yaxes(title_text="Impacto (R$)", row=2, col=2)
    fig_dashboard.update_yaxes(title_text="Valor", row=2, col=3)
    
    # Salvar como arquivo HTML
    fig_dashboard.write_html("dashboard_analise_detalhada.html")
    print(f"   📊 Dashboard salvo em: dashboard_analise_detalhada.html")
    
    # 3. GRÁFICO DE DISPERSÃO: Confiança vs Impacto
    fig_scatter = go.Figure()
    
    fig_scatter.add_trace(go.Scatter(
        x=df_detalhado['Confiança'],
        y=df_detalhado['Mudança_Percentual'],
        mode='markers+text',
        marker=dict(
            size=[abs(i)*2 + 10 for i in impacto_financeiro],
            color=df_detalhado['Variabilidade'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Variabilidade<br>(R$)"),
            line=dict(width=2, color='black')
        ),
        text=produtos_curtos,
        textposition="middle center",
        hovertemplate='<b>%{text}</b><br>' +
                     'Confiança: %{x:.1f}%<br>' +
                     'Mudança: %{y:+.1f}%<br>' +
                     'Impacto: R$ %{customdata:+.2f}<br>' +
                     '<extra></extra>',
        customdata=impacto_financeiro
    ))
    
    # Adicionar linhas de referência
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray", 
                         annotation_text="Sem mudança de preço")
    fig_scatter.add_vline(x=85, line_dash="dash", line_color="orange", 
                         annotation_text="Confiança mínima recomendada")
    
    fig_scatter.update_layout(
        title='Análise de Risco vs Retorno: Confiança × Mudança de Preços<br>' +
              '<sub>Tamanho das bolhas = impacto financeiro | Cor = variabilidade da predição</sub>',
        xaxis_title='Taxa de Confiança (%)',
        yaxis_title='Mudança Percentual no Preço (%)',
        height=600,
        width=1200,
        hovermode='closest'
    )
    
    # Salvar como arquivo HTML
    fig_scatter.write_html("analise_risco_retorno.html")
    print(f"   📊 Análise risco vs retorno salva em: analise_risco_retorno.html")
    
    print(f"\n{Colors.GREEN}✅ Visualizações interativas criadas com sucesso!{Colors.RESET}")
    print(f"   📊 3 gráficos HTML gerados:")
    print(f"      1. analise_precos_principal.html - Comparação detalhada Antes vs Depois")
    print(f"      2. dashboard_analise_detalhada.html - Dashboard de análise completa") 
    print(f"      3. analise_risco_retorno.html - Análise de risco vs retorno")
    print(f"\n{Colors.CYAN}💡 Para visualizar os gráficos, abra os arquivos HTML no seu navegador!{Colors.RESET}")

def documentar_limitacoes_modelo():
    """Documenta limitações e considerações importantes do modelo"""
    print_section("LIMITAÇÕES E CONSIDERAÇÕES DO MODELO")
    
    limitacoes = [
        "📊 LIMITAÇÕES DOS DADOS:",
        "   • Dados históricos podem não refletir tendências futuras",
        "   • Sazonalidade pode variar entre diferentes períodos",
        "   • Novos produtos sem histórico têm predições menos confiáveis",
        "",
        "🤖 LIMITAÇÕES DO MODELO:",
        "   • Modelo assume que padrões históricos continuarão válidos",
        "   • Features categóricas podem não capturar todas as nuances",
        "   • Outliers extremos podem afetar predições mesmo com tratamento",
        "",
        "💼 CONSIDERAÇÕES DE NEGÓCIO:",
        "   • Preços devem ser validados por especialistas de domínio",
        "   • Fatores externos (economia, concorrência) não são considerados",
        "   • Recomendações devem ser testadas em pequena escala primeiro",
        "",
        "🔄 RECOMENDAÇÕES DE USO:",
        "   • Re-treinar modelo mensalmente com dados atualizados",
        "   • Monitorar performance das predições em produção",
        "   • Combinar com análise de especialistas de pricing",
        "   • Usar intervalos de confiança para decisões de risco"
    ]
    
    for limitacao in limitacoes:
        print(f"   {limitacao}")
    
    logging.info("Limitações do modelo documentadas")

def mostrar_resumo_final(resultados, metricas_modelo):
    """Mostra resumo executivo final"""
    print_header("RESUMO EXECUTIVO FINAL")
    
    if not resultados:
        print(f"{Colors.RED}❌ Nenhum resultado para mostrar{Colors.RESET}")
        return
    
    df_final = pd.DataFrame(resultados)
    
    # KPIs principais
    print(f"{Colors.GREEN}📊 RESUMO GERAL:{Colors.RESET}")
    print(f"   🎯 Produtos analisados: {len(df_final)}")
    print(f"   📈 Confiança média: {df_final['Confiança_Predição'].mean():.1f}%")
    print(f"   💰 Preço médio recomendado: R$ {df_final['Preço_Recomendado'].mean():.2f}")
    print(f"   📊 R² do modelo: {metricas_modelo['r2_test']:.4f}")
    print(f"   🎯 MAPE: {metricas_modelo['mape_test']:.2f}%")
    
    # Tabela final
    print(f"\n{Colors.YELLOW}🏆 RESULTADOS FINAIS:{Colors.RESET}")
    print("─" * 95)
    print(f"{'PRODUTO':<40} | {'PREÇO REC.':<12} | {'CONFIANÇA':<10} | {'vs MÉDIA MERCADO':<15}")
    print("─" * 95)
    
    for _, row in df_final.iterrows():
        conf_icon = "🟢" if row['Confiança_Predição'] >= 90 else "🟡" if row['Confiança_Predição'] >= 80 else "🔴"
        diferenca = row['Preço_Recomendado'] - row['Preço_Médio_Mercado']
        
        produto_nome = row['Produto'][:37] + "..." if len(row['Produto']) > 37 else row['Produto']
        
        print(f"{conf_icon} {produto_nome:<37} | R$ {row['Preço_Recomendado']:8.2f} | "
              f"{row['Confiança_Predição']:6.1f}% | {diferenca:+8.2f}")
    
    print("─" * 95)
    
    # Análise de confiança
    alta_conf = len(df_final[df_final['Confiança_Predição'] >= 90])
    media_conf = len(df_final[(df_final['Confiança_Predição'] >= 80) & 
                             (df_final['Confiança_Predição'] < 90)])
    baixa_conf = len(df_final[df_final['Confiança_Predição'] < 80])
    
    print(f"\n{Colors.GREEN}🎯 ANÁLISE DE CONFIANÇA:{Colors.RESET}")
    print(f"   🟢 Alta confiança (≥90%): {alta_conf} produtos")
    print(f"   🟡 Média confiança (80-89%): {media_conf} produtos")
    print(f"   🔴 Baixa confiança (<80%): {baixa_conf} produtos")
    
    # Top 10 recomendações
    top_10 = df_final.nlargest(10, 'Confiança_Predição')
    print(f"\n{Colors.CYAN}🏆 TOP 10 RECOMENDAÇÕES (por confiança):{Colors.RESET}")
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        print(f"   {i}. {row['Produto']}")
        print(f"      💰 Preço: R$ {row['Preço_Recomendado']:.2f}")
        print(f"      🎯 Confiança: {row['Confiança_Predição']:.1f}%")
        print(f"      📊 Diferença vs mercado: {row['Diferença_vs_Média']:+.2f} R$")
        if row['Justificativas']:
            print(f"      📋 Justificativa: {row['Justificativas'][0]}")
        print()

# ============================================================================
# 7. FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal do sistema"""
    print_header("SISTEMA DE RECOMENDAÇÃO DE PREÇOS - OLIST E-COMMERCE")
    
    # Configurar caminho dos dados (relativo ao diretório do script)
    caminho_dados = os.path.dirname(os.path.abspath(__file__))
    
    print(f"{Colors.BLUE}📁 Caminho dos dados: {caminho_dados}{Colors.RESET}")
    
    # Verificar se o caminho existe
    if not os.path.exists(caminho_dados):
        print(f"{Colors.RED}❌ Caminho não encontrado: {caminho_dados}{Colors.RESET}")
        print(f"{Colors.YELLOW}💡 Verifique se o caminho está correto e tente novamente{Colors.RESET}")
        return
    
    try:
        # 1. Carregar dados
        datasets = carregar_dados(caminho_dados)
        if not datasets:
            return
        
        # 2. Juntar datasets
        df_main = juntar_datasets(datasets)
        
        # 3. Limpar dados
        df_clean = limpar_dados(df_main)
        
        # 4. Engenharia de features
        df_features, le_category, le_customer_state, le_seller_state = engenharia_features(df_clean)
        
        # 5. Treinar modelos (agora com múltiplos algoritmos)
        resultado_completo = treinar_modelos_multiplos(df_features)
        modelo, scaler, features_disponiveis, feature_importance, metricas = resultado_completo[:5]
        
        # 5.1. Salvar dados de referência e modelo para monitoramento
        salvar_dados_referencia(df_features[features_disponiveis])
        
        # Inicializar sistema de retreinamento
        retreinamento = ModeloRetreinamento()
        retreinamento.salvar_modelo(
            modelo, scaler, features_disponiveis, 
            metadata={'metricas': metricas, 'feature_importance': feature_importance.to_dict() if hasattr(feature_importance, 'to_dict') else {}}
        )
        
        # 6. Analisar produtos e gerar recomendações
        resultados = analisar_produtos(df_features, modelo, scaler, 
                                     features_disponiveis, feature_importance)
        
        # 7. Criar visualizações
        criar_visualizacoes(resultados, metricas)
        
        # 8. Documentar limitações
        documentar_limitacoes_modelo()
        
        # 9. Mostrar resumo final
        mostrar_resumo_final(resultados, metricas)
        
        print_header("ANÁLISE CONCLUÍDA COM SUCESSO!")
        
    except Exception as e:
        print(f"{Colors.RED}❌ Erro durante a execução: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()