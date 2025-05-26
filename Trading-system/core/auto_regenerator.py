import optuna
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from typing import Tuple, Any
import pickle
import os

class AutoRegenerator:

# ----------------------------------------------------------------- #
    def __init__(self):
        # Cria um estudo do Optuna para minimizar o erro
        self.study = optuna.create_study(direction='minimize')
        self.best_model = None  # Modelo com melhor desempenho
        self.best_score = np.inf  # Melhor valor de erro (quanto menor, melhor)
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
    def _objective(self, trial, X: np.ndarray, y: np.ndarray) -> float:
        # Função objetivo que será chamada pelo Optuna para otimizar os hiperparâmetros

        model_type = trial.suggest_categorical('model_type', ['xgb', 'rf'])

        if model_type == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            model = XGBRegressor(**params)
        else:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
            }
            model = RandomForestRegressor(**params)

        # Validação cruzada simples com 3 execuções
        scores = []
        for _ in range(3):
            idx = np.random.permutation(len(X))
            split = int(0.8 * len(X))
            train_idx, val_idx = idx[:split], idx[split:]
            
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[val_idx])
            scores.append(mean_squared_error(y[val_idx], preds))

        return np.mean(scores)
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
    def optimize(self, X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> Tuple[Any, float]:
        # Executa o processo de otimização de hiperparâmetros
        
        self.study.optimize(lambda trial: self._objective(trial, X, y), n_trials=n_trials)
        
        # Recupera os melhores hiperparâmetros encontrados
        best_params = self.study.best_params

        # Treina o melhor modelo nos dados completos
        if best_params['model_type'] == 'xgb':
            self.best_model = XGBRegressor(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                learning_rate=best_params['learning_rate']
            )
        else:
            self.best_model = RandomForestRegressor(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth']
            )
        
        self.best_model.fit(X, y)
        self.best_score = self.study.best_value

        # Garante que o diretório de modelos existe
        os.makedirs('models', exist_ok=True)

        # Salva o modelo treinado em disco
        with open('models/best_autoreg.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)

        return self.best_model, self.best_score
# ----------------------------------------------------------------- #

# ----------------------------------------------------------------- #
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Realiza previsão com o modelo treinado
        if self.best_model is None:
            raise ValueError("Modelo ainda não foi treinado")
        return self.best_model.predict(X)
# ----------------------------------------------------------------- #
