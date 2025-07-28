import os
import threading
from multiprocessing import Queue, Event
import numpy as np
from llama_cpp import Llama, ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage

from .animations import load_animations
from .body_model import BodyModel

PROMPT = \
    """
    The user made a gesture that you interpret as {user_gesture}.
    The user describes a gesture. You need to creatively respond with exactly ONE of the following options:
    {options}
    Just give the one word and nothing else!!
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
                repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
                filename="Llama-3.2-3B-Instruct-Q8_0.gguf",
                n_gpu_layers=-1,  # Use all layers on GPU
            )

            while True:
                value = self.input_queue.get()
                print("Received input: ", value)
                array = value.landmarks
                if np.any(array):
                    vec = BodyModel(data=array).get_angle_vector()
                    closest_match = self.find_most_similar_vector(vec, similarity='cosine')
                    print(f"Closest match: {closest_match}")


                    prompt:ChatCompletionRequestSystemMessage = {"role": "system", "content": PROMPT.format(options=", ".join(self.options), user_gesture=closest_match)}

                    print("Starting Completion")
                    response = llm.create_chat_completion(
                        top_p=0.95,
                        temperature=0.7,
                        max_tokens=50,
                        messages=[ prompt ]
                    )

                    result = response["choices"][0]["message"]["content"]
                    print(f"LBLM Raw Response: {result}")
                    words = [word.strip() for word in result.split(',') if word.strip() and word.strip() in self.options]
                    print(f"LBLM Filtered Response: {words}")
                    for word in words:
                        self.output_queue.put(word)
        except Exception as e:
            print(f"Brain Freeze: {e}")
