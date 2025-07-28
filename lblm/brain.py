import os
import threading
from multiprocessing import Queue, Event

import lmstudio as lms
import numpy as np

from animations import load_animations
from model import BodyModel

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
        self.options = load_animations(base_path="animations", animation_folder="GLBs")
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
    def load_vectors():
        """
        Loads all saved vectors from the saved vectors folder.
        :return:
        """
        arrays = {}
        for filename in os.listdir("animations/vectors"):
            if filename.endswith('.npy'):
                key = os.path.splitext(filename)[0]
                full_path = os.path.join("animations/vectors", filename)
                array = np.load(full_path, allow_pickle=True)
                arrays[key] = BodyModel(data=array).get_angle_vector()
        print(f"Loaded {len(arrays.keys())} vectors")
        print(arrays["idle"])
        return arrays

    def run(self):
        self.is_outputting_event.set()
        model = lms.llm("google/gemma-3-4b")
        while True:
            value = self.input_queue.get()
            print("Received input: ", value)
            array = value.landmarks
            if np.any(array):
                vec = BodyModel(data=array).get_angle_vector()
                closest_match = self.find_most_similar_vector(vec, similarity='cosine')
                print(f"Closest match: {closest_match}")
                chat = lms.Chat(self.prompt)
                chat.add_user_message("The user made a gesture that you interpret with: " + closest_match)
                response = model.respond(chat)
                result = response.content
                words = [word.strip() for word in result.split(',') if word.strip() and word.strip() in self.options]
                for word in words:
                    self.output_queue.put(word)
