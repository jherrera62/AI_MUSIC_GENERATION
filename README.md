#Music Generation with Tensorflow
This program utilizes Tensorflow to generate music by taking MIDI files as input and producing new MIDI files as output. The project is documented in the generate.py file, which contains the code used to create this music generation system.

#Usage
To use this program, follow these simple steps:

Make sure you have the necessary dependencies installed.

Run the program by executing the command python3 generate.py in your terminal.

Once the program is running, it will start reading MIDI files from the midis folder as input.

The MIDI files will be converted into a state matrix, which will serve as the input for the neural network.

The neural network will generate a new matrix based on the provided input.

The generated matrix will be converted back into a MIDI file.

The resulting MIDI file will be saved in the generated_songs folder.

#Dependencies
Before running the program, ensure that you have the following dependencies installed:

Tensorflow
Cuda
TensorRT
numpy
mido
tqdm
opencv-python
libcunn8

#References
Please refer to the generate.py file for a list of references and resources used in the development of this music generation project.

Feel free to explore the code and experiment with different MIDI files to generate unique and original music compositions.

Enjoy the music generation experience!
