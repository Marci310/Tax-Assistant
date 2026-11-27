import os
import random

from locust import HttpUser, between, task

ASK = "/api/v1/ask"
TAX_QUESTIONS = [
    "Mi a növekedési adóhitel?",
    "Mikor kell benyújtani a társasági adó bevallását?",
    "Ki lehet a Tao-felajánlás kedvezményezettje?",
    "Milyen adókötelezettségei vannak egy nonprofit szervezetnek?",
    "What is the corporate tax rate?",
    "Who is subject to corporate income tax?",
]
OFF_TOPIC_QUESTIONS = [
    "Hogyan kell pizzát sütni?",
    "Who won the 2022 football World Cup?",
]


class TaxAssistantUser(HttpUser):
    host = os.environ.get("BASE_URL", "http://localhost:8000")
    wait_time = between(1, 3)
    network_timeout = 90.0

    def on_start(self):
        self.client.headers["X-API-Key"] = os.environ["API_KEY"]

    @task(4)
    def ask_tax_question(self):
        self.client.post(ASK, json={"question": random.choice(TAX_QUESTIONS)})

    @task(1)
    def ask_off_topic_question(self):
        self.client.post(ASK, json={"question": random.choice(OFF_TOPIC_QUESTIONS)})
