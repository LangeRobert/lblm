import threading
from queue import Queue
from typing import List

import lmstudio as lms


class Controller(threading.Thread):
    def __init__(self, output_queue, options:List[str]):
        super().__init__()
        self.output_queue:Queue = output_queue
        self.options = options

        self.prompt = f"""
            Just respond with a combination of the following words:
            {','.join(self.options)}
            Separate words with a comma. Dont use any other text.
            Just use 3-5 words to describe what you want to say.
        """

    def run(self):
        model = lms.llm("google/gemma-3-4b")
        chat = lms.Chat(self.prompt)
        while True:
            user_input = input()
            chat.add_user_message(user_input)
            response = model.respond(chat)
            self.output_queue.put(response)



if __name__ == '__main__':
    output_queue = Queue()
    options = ['angry', 'wave', 'cheer', 'play_guitar', 'walk_back', 'punch', 'hit_baseball', 'dance', 'joy_jump', 'die', 'hang']
    controller = Controller(output_queue, options)
    controller.start()

    # Example of how to use the output queue
    while True:
        if not output_queue.empty():
            message = output_queue.get()
            print(f"{message}", end="\n\n ---")