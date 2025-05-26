import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from typing import Tuple

# --------------------------------------------------------- #
class TradingModel:
    def __init__(self, input_shape: Tuple[int, int], action_space: int):
        self.input_shape = input_shape  # Formato da entrada: (janela de tempo, número de features)
        self.action_space = action_space  # Número de ações possíveis (por exemplo: comprar, vender, manter)
        self.model = self._build_model()  # Modelo principal
        self.target_model = self._build_model()  # Modelo alvo (para estabilidade no treinamento)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Otimizador Adam
        self.loss_fn = tf.keras.losses.Huber()  # Função de perda Huber (robusta a outliers)
        self.update_target()  # Inicializa os pesos do modelo alvo
# --------------------------------------------------------- #

# --------------------------------------------------------- #
    def _build_model(self) -> models.Model:
        # Define a arquitetura da rede neural
        inputs = layers.Input(shape=self.input_shape)
        
        # Processamento temporal com LSTM
        lstm_out = layers.LSTM(64, return_sequences=True)(inputs)
        
        # Mecanismo de atenção sobre as saídas da LSTM
        attention = layers.Attention()([lstm_out, lstm_out])
        
        # Combina as saídas da atenção e da LSTM
        x = layers.Concatenate()([layers.Flatten()(attention), layers.Flatten()(lstm_out)])
        x = layers.Dense(128, activation='relu')(x)  # Camada densa com ativação ReLU
        x = layers.Dropout(0.3)(x)  # Regularização com Dropout
        
        # Camada de saída com valores Q para cada ação possível
        outputs = layers.Dense(self.action_space)(x)
        
        # Cria e retorna o modelo
        return models.Model(inputs=inputs, outputs=outputs)
# --------------------------------------------------------- #

# --------------------------------------------------------- #  
    def update_target(self):
        # Atualiza os pesos do modelo alvo com os pesos atuais do modelo principal
        self.target_model.set_weights(self.model.get_weights())
# --------------------------------------------------------- #

# --------------------------------------------------------- #  
    def train(self, states: np.ndarray, actions: np.ndarray, 
              rewards: np.ndarray, next_states: np.ndarray, 
              dones: np.ndarray, gamma: float = 0.95) -> float:
        
        # Calcula os Q-valores alvo com base nas próximas observações
        next_q = self.target_model.predict(next_states, verbose=0)
        target_q = rewards + (1 - dones) * gamma * np.amax(next_q, axis=1)
        
        # Etapa de treinamento
        with tf.GradientTape() as tape:
            current_q = self.model(states)
            action_masks = tf.one_hot(actions, self.action_space)  # Cria máscara one-hot para ações tomadas
            q_values = tf.reduce_sum(current_q * action_masks, axis=1)  # Extrai os Q-valores das ações tomadas
            loss = self.loss_fn(target_q, q_values)  # Calcula a perda entre Q atual e Q alvo
        
        # Calcula e aplica os gradientes
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return float(loss.numpy())  # Retorna o valor da perda
# --------------------------------------------------------- #

# --------------------------------------------------------- #
    def predict(self, state: np.ndarray) -> np.ndarray:
        # Faz uma previsão com o modelo dado um único estado
        return self.model.predict(state[np.newaxis, ...], verbose=0)[0]
# --------------------------------------------------------- #

# --------------------------------------------------------- #
    def save(self, path: str):
        # Salva o modelo principal no caminho especificado
        self.model.save(path)
# --------------------------------------------------------- #

# --------------------------------------------------------- #
    def load(self, path: str):
        # Carrega o modelo salvo e atualiza o modelo alvo
        self.model = models.load_model(path)
        self.update_target()
# --------------------------------------------------------- #
