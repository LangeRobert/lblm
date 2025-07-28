import os
import threading
from multiprocessing import Queue, Event
import numpy as np
from llama_cpp import Llama, ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage

from .animations import load_animations
from .body_model import BodyModel

PROMPT = \
    """
    Respond in just one or two words, using the following options:
    {options}
    Separate options with a comma. Dont use any other text. Try to use just one pantomime if possible. Be very creative.
    """


class Brain(threading.Thread):
    def __init__(self):
        super().__init__()
        # communication queues
        self.input_queue = Queue()
        self.output_queue = Queue()

        # events
        self.stop_event = Event()
        self.is_outputting_event = Event()

        # options
        self.options = load_animations(animations_path="lblm/data/animations")
        self.vectors = self.load_vectors()

        self.prompt = PROMPT.format(options=", ".join(self.options))

    def find_most_similar_vector(self, query_vector, similarity='cosine'):
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        best_key = None
        best_score = -np.inf

        for key, vec in self.vectors.items():
            if similarity == 'cosine':
                score = cosine_similarity(query_vector, vec)
            elif similarity == 'dot':
                score = np.dot(query_vector, vec)
            else:
                raise ValueError("Unsupported similarity metric")

            if score > best_score:
                best_score = score
                best_key = key

        return best_key

    @staticmethod
    def load_vectors(path="lblm/data/search_space"):
        """
        Loads all saved vectors from the saved vectors folder.
        :return:
        """
        arrays = {}
        for filename in os.listdir(path):
            if filename.endswith('.npy'):
                key = os.path.splitext(filename)[0]
                full_path = os.path.join(path, filename)
                array = np.load(full_path, allow_pickle=True)
                arrays[key] = BodyModel(data=array).get_angle_vector()
        print(f"Loaded {len(arrays.keys())} vectors")
        print(arrays["idle"])
        return arrays

    def run(self):
        try:
            self.is_outputting_event.set()

            llm = Llama.from_pretrained(
                repo_id="unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF",
                filename="DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf",
            )

            while True:
                value = self.input_queue.get()
                print("Received input: ", value)
                array = value.landmarks
                if np.any(array):
                    vec = BodyModel(data=array).get_angle_vector()
                    closest_match = self.find_most_similar_vector(vec, similarity='cosine')
                    print(f"Closest match: {closest_match}")


                    system_prompt:ChatCompletionRequestSystemMessage = {"role": "system", "content": self.prompt.format(options=", ".join(self.options))}
                    user_message:ChatCompletionRequestUserMessage = {"role": "user", "content": "The user made a gesture that you interpret with: " + closest_match}

                    print("Starting Completion")
                    response = llm.create_chat_completion(
                        top_p=0.95,
                        temperature=0.7,
                        max_tokens=50,
                        messages=[
                            system_prompt,
                            user_message
                        ]
                    )
                    result = response.choices[0].message.content
                    print(f"LBLM Response: {result}")
                    words = [word.strip() for word in result.split(',') if word.strip() and word.strip() in self.options]
                    for word in words:
                        self.output_queue.put(word)
        except Exception as _:
            print(f"Brain Freeze")
