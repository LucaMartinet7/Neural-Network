# Neural Network Project

Introduction
============

This project, named `Neural Network`, implements a neural network for analyzing chess game states. It includes tools for generating, training, and predicting neural networks based on chessboard configurations in FEN notation.

Project Structure
=================

Configuration
=============

The `basic_network.conf` file contains the configuration for the neural network:

```conf
input_layer.input_neurons = 832
output_layer.output_neurons = 6
hidden_layer_1.neurons = 128
hidden_layer_2.neurons = 64
training.learning_rate = 0.001
training.epochs = 16
```

Usage
=====

Generating the Network
----------------------
```bash
./my_torch_generator basic_network.conf
```

Training the Network
---------------------
```bash
./my_torch_analyzer --train trained_network_1.nn dataset.txt --save trained_network_1_trained.nn
```

Making Predictions
-------------------
```bash
./my_torch_analyzer --predict trained_network_1_trained.nn fen_input.txt
```

Authors
=======

Group: Ratio

- [Luca Martinet](https://github.com/Lucamartinet7)
- [Noe Kurata](https://github.com/nkurata)

Note for Epitech Students
=========================
Using this repository for your Epitech coursework will result in a -42. This repository contains all my projects completed during my 5th semester at Epitech in 2024. Please refrain from submitting any content from this repository as your own work.

License
=======
This repository is provided for informational and educational purposes only. All rights to the original projects belong to the author.

Support
=======
If you are having issues, please let us know. Contact me via the email on my GitHub profile.