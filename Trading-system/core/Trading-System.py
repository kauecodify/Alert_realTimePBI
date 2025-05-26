import numpy as np
from typing import List, Dict
from datetime import datetime
import random
import json
from .data_processor import DataProcessor
from .trading_model import TradingModel
from .auto_regenerator import AutoRegenerator
import pickle
import os

class TradingSystem:

# ------------------------------------------------------------- #
    def __init__(self):
        # Inicializa os componentes do sistema
        self.data_processor = DataProcessor()
        self.model = TradingModel(
            input_shape=(60, 4),  # janela de tempo x número de features
            action_space=3        # ações: comprar, manter, vender
        )
        self.auto_ml = AutoRegenerator()  # Otimização com AutoML
        self.memory = []  # Memória para experiência (replay buffer)

        # Portfólio inicial > (wallet analyst)
        self.portfolio = {
            'cash': 100000,
            'shares': 0,
            'total_value': 100000
        }

        self.window_size = 60  # Janela de observação
        self.step_count = 0    # Contador de passos

# ------------------------------------------------------------- #
    def run(self):
        # Loop principal que consome dados e processa em tempo real
        data_window = []
        for data in self.data_processor.stream_data():
            data_window.append(data)

            # Mantém a janela com o tamanho desejado
            if len(data_window) > self.window_size * 2:
                data_window = data_window[-self.window_size * 2:]

            # Processa quando há dados suficientes
            if len(data_window) >= self.window_size:
                self._process_window(data_window)

            # Periodicamente re-treina os modelos AutoML
            if self.step_count % 1000 == 0:
                self._retrain_models()

            self.step_count += 1
          
# ------------------------------------------------------------- #
    def _process_window(self, window: List[Dict]):
        # Gera features e executa decisões
        features = self.data_processor.create_features(window)
        if len(features) < 2:
            return

        current_state = features[-1]
        prev_state = features[-2]

        # Seleciona a ação com base no estado atual
        action = self._select_action(current_state)

        # Executa ação e calcula recompensa
        reward = self._execute_action(action, window[-1]['price'])

        # Armazena experiência no replay buffer
        self.memory.append((prev_state, action, reward, current_state, False))

        # Treina o modelo periodicamente
        if len(self.memory) >= 1000 and self.step_count % 100 == 0:
            self._train_model()
          
# ------------------------------------------------------------- #
    def _select_action(self, state: np.ndarray) -> int:
        # Estratégia epsilon-greedy
        epsilon = max(0.1, 1 - self.step_count / 10000)
        if random.random() < epsilon:
            return random.randint(0, 2)  # Escolha aleatória
        else:
            return np.argmax(self.model.predict(state))  # Escolha baseada no modelo
# ------------------------------------------------------------- #
    def _execute_action(self, action: int, price: float) -> float:
        # Executa a ação escolhida e atualiza o portfólio
        old_value = self.portfolio['total_value']

        if action == 0:  # Comprar
            shares = self.portfolio['cash'] // price
            self.portfolio['cash'] -= shares * price
            self.portfolio['shares'] += shares
        elif action == 2:  # Vender
            self.portfolio['cash'] += self.portfolio['shares'] * price
            self.portfolio['shares'] = 0

        self.portfolio['total_value'] = self.portfolio['cash'] + self.portfolio['shares'] * price

        # Retorna a recompensa (variação percentual do portfólio)
        return (self.portfolio['total_value'] - old_value) / old_value
      
# ------------------------------------------------------------- #
    def _train_model(self):
        # Treinamento com uma amostra aleatória da memória
        batch = random.sample(self.memory, min(64, len(self.memory)))
        states, actions, rewards, next_states, dones = zip(*batch)

        loss = self.model.train(
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

        # Atualiza a rede-alvo periodicamente
        if self.step_count % 1000 == 0:
            self.model.update_target()
          
# ------------------------------------------------------------- #
    def _retrain_models(self):
        # Re-treinamento com AutoML se houver dados suficientes
        if len(self.memory) < 1000:
            return

        # Prepara dados supervisionados para AutoML
        X = np.array([x[0] for x in self.memory])
        y = np.array([x[2] for x in self.memory])  # recompensas como target

        # Otimiza novo modelo
        new_model, score = self.auto_ml.optimize(X, y)

        # Substitui modelo se houver melhoria significativa
        if score < self.auto_ml.best_score * 0.9:  # Melhoria de 10%
            self._integrate_new_model(new_model)
          
# ------------------------------------------------------------- #
    def _integrate_new_model(self, new_model):
        # Aqui poderia ser feita a substituição parcial do modelo RL
        # salva o novo modelo AutoML
        os.makedirs('models', exist_ok=True)
        with open('models/new_autoreg_model.pkl', 'wb') as f:
            pickle.dump(new_model, f)
          # ------------------------------------------------------------- #
