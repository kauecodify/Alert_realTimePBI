<img src="https://github.com/user-attachments/assets/6e845136-dd47-4060-b172-72e954c63777/" width="150px" alt="Brasil" />

### Sistema de Alertas com Orquestração no Power BI em Tempo Real

- INFRA

<img src="https://img.icons8.com/?size=100&id=cvzmaEA4kC0o&format=png&color=000000" />
Kubernetes cluster (para escalabilidade)

<img src="https://img.icons8.com/?size=100&id=fOhLNqGJsUbJ&format=png&color=000000" />
Servidor Kafka (stream de dados)

<img src="https://img.icons8.com/?size=100&id=qYfwpsRXEcpc&format=png&color=000000"/>
Power BI (dashboards em tempo real)


# ----------------------------- #
- BIBLIOTECAS

pip install numpy pandas tensorflow kafka-python pyarrow optuna scikit-learn xgboost kubernetes prometheus_client python-dotenv pytz tensorflow-addons pickle5


# ----------------------------- #
- Configuração

Arquivo .env:

KAFKA_BROKERS=kafka-cluster:9092

KAFKA_TOPICS=stock-prices,market-news

POWERBI_CLIENT_ID=your-client-id

POWERBI_TENANT_ID=your-tenant-id

POWERBI_DATASET_ID=your-dataset-id


# ----------------------------- #
- Deploy no Kubernetes:

bash
kubectl apply -f deployment/


# ----------------------------- #
- Estrutura do Projeto

![image](https://github.com/user-attachments/assets/7087e03c-acc5-46a6-8472-67a03beaaf36)


# ----------------------------- #
- Como Executar

Iniciar o sistema principal:
bash
python -m core.trading_system


# ----------------------------- #
- Monitoramento:
- 
Métricas: http://localhost:8000

Dashboard: http://localhost:3000 (Grafana)


# ----------------------------- #
- Power BI:
Importar arquivos de powerbi e Configurar conexão com o dataset
