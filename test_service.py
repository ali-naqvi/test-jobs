from ray import serve
from fastapi import FastAPI
import os

msg = os.getenv("MSG", "ok")

serve.start(detached=True)
app = FastAPI()

@serve.deployment(route_prefix="/")
@serve.ingress(app)
class App:
    def __init__(self):
        self.should_fail = False

    @app.get("/healthcheck")
    def healthcheck(self):
        if self.should_fail:
            raise RuntimeError("Oh no")
        return msg

    @app.post("/set_should_fail")
    def set_should_fail(self):
        self.should_fail = True
        return "ok"
App.deploy()

