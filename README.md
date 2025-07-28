# Large Body Language Model

The Large Body Language Model is an interactive installation made in the context of a seminar at the 
Einstein Center Digital Future in Berlin.

# Run the Installation
### 0. Requirements
- Python 3.10 or higher
- A dedicated GPU (Cuda is recommended, but Apple Silicon is also supported)
- Webcam
- A Display or Projector
- roughly 10 GB of free disk space / min 16 GB RAM

### 1. Download the repository
You can download the repository manually or by using git:
```bash
git clone https://github.com/LangeRobert/lblm
```

### 2. Install the requirements
All required Python packages can be installed using pip. 
Make sure you are in the root directory of the cloned repository and run:

```bash
pip install -r requirements.txt
```

### 3. Start
To start the installation, run the following command in the root directory of the cloned repository:
```bash
python -m lblm
```
The first execution will take some time, as the model will be downloaded and cached.
After that, the installation will start and you will see a window with the webcam feed and the displayed animation.

### 4. Interact
You can interact with the installation by moving your body in front of the webcam.
Try things out. Have fun. Think about the first interactions with 