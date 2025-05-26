import pyarrow as pa
import pandas as pd
from kafka import KafkaConsumer
import numpy as np
from typing import Dict, List, Generator
import os
import pickle 
import json  # deserializar mensagens do Kafka
from datetime import datetime

class DataProcessor:
    def __init__(self):
        # Define o esquema de dados esperado
        self.schema = self._create_schema()
        # Garante que os diretórios existem
        self._setup_directories()

    def _create_schema(self) -> pa.Schema:
        # Define o esquema dos dados usando PyArrow
        return pa.schema([
            ('timestamp', pa.timestamp('ns')),
            ('symbol', pa.string()),
            ('price', pa.float64()),
            ('volume', pa.int64()),
            ('open', pa.float64()),
            ('high', pa.float64()),
            ('low', pa.float64()),
            ('close', pa.float64())
        ])

    def _setup_directories(self):
        # Cria os diretórios para salvar os dados brutos e processados, se ainda não existirem
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)

    def stream_data(self) -> Generator[Dict, None, None]:
        # Conecta ao Kafka para consumir dados do tópico 'stock-prices'
        consumer = KafkaConsumer(
            'stock-prices',
            bootstrap_servers='localhost:9092',
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='latest'
        )
        
        # Itera sobre as mensagens recebidas
        for message in consumer:
            try:
                data = self._validate_data(message.value)
                if data:
                    yield data  # Retorna os dados válidos
            except Exception as e:
                print(f"Erro no processamento dos dados: {str(e)}")

    def _validate_data(self, data: Dict) -> Dict:
        # Verifica se os campos obrigatórios estão presentes
        required = ['symbol', 'price', 'timestamp']
        if not all(field in data for field in required):
            return None
        
        try:
            # Converte o timestamp e o preço para os formatos corretos
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data['price'] = float(data['price'])
            return data
        except:
            return None  # Retorna None se a conversão falhar

    def create_features(self, window: List[Dict]) -> np.ndarray:
        # Cria um DataFrame com a janela de dados recebida
        df = pd.DataFrame(window)
        features = []

        # Médias móveis simples
        df['sma_5'] = df['price'].rolling(5).mean()
        df['sma_20'] = df['price'].rolling(20).mean()
        
        # Cálculo do RSI (Índice de Força Relativa)
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        
        # Normalização dos preços
        df['norm_price'] = (df['price'] - df['price'].mean()) / df['price'].std()
        
        # Retorna os recursos como array NumPy
        return df[['sma_5', 'sma_20', 'rsi', 'norm_price']].values

    def save_batch(self, data: List[Dict], batch_type: str):
        # Salva um lote de dados em um arquivo pickle
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        with open(f'data/{batch_type}/batch_{timestamp}.pkl', 'wb') as f:
            pickle.dump(data, f)  ## add logs
