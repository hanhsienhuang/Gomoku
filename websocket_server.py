#!/usr/bin/env python

# WS server example

import asyncio
import websockets
import utils
import agent
import time

async def hello(websocket, path):
    print(path)
    name = await websocket.recv()
    print(f"< {name}")

    greeting = f"Hello {name}!"
    time.sleep(60)

    await websocket.send(greeting)
    print(f"> {greeting}")


async def serve_game(websocket, path):
    size = await websocket.recv()
    wsAgent = agent.WebsocketAgent(websocket)
    baselineAgent = agent.BaselineAgent0()
    #await utils.async_play(wsAgent, baselineAgent, True)
    await utils.async_play(baselineAgent, wsAgent, True)
    #await utils.async_play(wsAgent, wsAgent, True)

#start_server = websockets.serve(hello, "localhost", 8765, ping_interval=None)
start_server = websockets.serve(serve_game, "localhost", 8765, ping_interval=None)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
