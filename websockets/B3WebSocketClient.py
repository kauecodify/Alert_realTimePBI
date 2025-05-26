import websockets
from authlib.integrations.httpx_client import OAuth2Client

class B3WebSocketClient:
    def __init__(self, token):
        self.uri = "wss://api.b3.com.br/websocket"
        self.token = token
        
    async def connect(self):
        async with websockets.connect(
            self.uri,
            extra_headers={"Authorization": f"Bearer {self.token}"}
        ) as ws:
            # Subscreve aos ativos desejados
            await ws.send(json.dumps({
                "action": "subscribe",
                "symbols": ["PETR4", "VALE3"]
            }))
            
            while True:
                data = await ws.recv()
                print("Dados recebidos:", json.loads(data))
