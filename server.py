import asyncio
import json
import websockets

clients = {
    'sender': None,
    'receiver': None,
    'sender1': None
}

async def handler(websocket, path):
    if path == "/sender":
        clients['sender'] = websocket
    elif path == "/receiver":
        clients['receiver'] = websocket
    elif path == "/sender1":
        clients['sender1'] = websocket

    async for message in websocket:
        data = json.loads(message)
        print(f"Received data from {path}: {data}")

        # Send received data to the receiver client
        if clients['receiver']:
            response = json.dumps({"received_from_sender": data})
            await clients['receiver'].send(response)
            print("Sent data to receiver")

        # Send acknowledgment to the sender client
        if clients['sender'] or clients['sender1']:
            ack = json.dumps({"response": "Data received"})
            await clients['sender'].send(ack)
            print("Sent acknowledgment to sender")

start_server = websockets.serve(handler, "localhost", 8765)

async def main():
    await start_server
    print("\033[92mServer started and ready at ws://localhost:8765\033[0m")
    await asyncio.Future()  # Run forever

asyncio.get_event_loop().run_until_complete(main())