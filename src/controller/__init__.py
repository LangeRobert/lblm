import threading
from queue import Queue
from typing import List
import lmstudio as lms

PROMPT = \
"""
Respond in just one or two words, using the following options:
{options}
Separate options with a comma. Dont use any other text. Try to use just one pantomime if possible.
"""

class Controller(threading.Thread):
    def __init__(self, output_queue, options:List[str]):
        super().__init__()
        self.output_queue:Queue = output_queue
        self.options = [o for o in options if not o == "idle"]

        self.prompt = PROMPT.format(options=", ".join(self.options))

    def run(self):
        model = lms.llm("google/gemma-3-4b")
        while True:
            user_input = input()
            chat = lms.Chat(self.prompt)
            chat.add_user_message(user_input)
            response = model.respond(chat)
            result = response.content
            print(f"Response: {result}")

            # Split the response into words and filter out empty strings
            words = [word.strip() for word in result.split(',') if word.strip() and word.strip() in self.options]
            # Ensure the response contains only valid options
            for word in words:
                self.output_queue.put(word)