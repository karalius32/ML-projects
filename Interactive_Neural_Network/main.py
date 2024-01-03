import pygame
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
import spiral_data
import numpy as np
from pygame_widgets.button import Button
from pygame_widgets.dropdown import Dropdown
import pygame_widgets
import neural_network as nn
from torch import nn as torch_nn
import torch


screen_width = 1280
screen_height = 940

pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()
running = True
model_type = "Custom"

btn_colors = ((255, 255, 255), (230, 230, 230), (205, 205, 205), (180, 180, 180))

def draw_weights(positions):
        for i in range(len(positions) - 1):
            for j in range(len(positions[i])):
                for k in range(len(positions[i + 1])):
                    pygame.draw.line(screen, "black", positions[i][j], positions[i + 1][k], 3)

def plot(fig_top_left, width, data, c, cmap, model_type):
    fx, fy = fig_top_left
    pygame.draw.rect(screen, "black", (fx, fy, width, width), 3)
    for d, _c in zip(data, c):
        color = cmap[_c]
        if model_type == "PyTorch":
            d_0 = d[0].item()
            d_1 = d[1].item()
        else:
            d_0 = d[0]
            d_1 = d[1]
        pygame.draw.circle(screen, color, (fx + d_0 * (width - 8), fy + d_1 * (width - 8)), 4)

def plot_prediction(fig_top_left, width, model, cmap):
    xx, yy = np.meshgrid(np.linspace(fig_top_left[0], fig_top_left[0] - 5 + width, 200), np.linspace(fig_top_left[1], fig_top_left[1] - 5 + width, 200))
    X = np.c_[(xx.ravel() - fig_top_left[0]) / width, (yy.ravel() - fig_top_left[1]) / width]
    if model_type == "Custom":
        pred = model.predict(X)
    else:
        X = torch.tensor(X).to(torch.float32)
        pred = model(X).detach()
    pred = np.argmax(pred, axis=1)
    for x, y, pr in zip(xx.ravel(), yy.ravel(), pred):
        color = cmap[pr]
        pygame.draw.rect(screen, color, (x, y, 4, 4))

def start_fit(layers):
    global model, fit_model, total_epochs, model_type, optimizer, learning_rate
    model_type = dropdown_backend.getSelected()
    if model_type == None: model_type = "Custom"
    if model_type == "PyTorch":
        model = nn.ModelTorch(layers)
        optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        model = nn.Model(layers)
    dataset.to_tensor(model_type)
    fit_model = True
    total_epochs = 0

def stop_fit():
    global fit_model
    fit_model = False


class Dataset():
    def __init__(self, model_type):
        self.seed = np.random.choice(range(9999), 1)[0]
        self.dataset_name = "moons"
        self.X, self.y = make_moons(100, noise=0.1)
        self.X[:, 0] *= 0.5
        self.X = self.X + np.abs(np.min(self.X))
        self.X = self.X / np.max(self.X)
        self.X = np.clip(0.03, self.X, np.max(self.X) - 0.03)
        self.classes_n = 2
        self.y_encoded = np.array([[1 if j == i else 0 for j in range(self.classes_n)] for i in self.y])
        if model_type == "PyTorch":
            self.X = torch.tensor(self.X).to(torch.float32)
            self.y = torch.tensor(self.y)

    def load_data(self, dataset_name, add_class=0):
        stop_fit()
        if (dataset_name == "moons" or dataset_name == "xor") and add_class != 0:
            return
        if self.classes_n + add_class == 1 or self.classes_n + add_class == 5:
            return
        if add_class == 0:
            self.seed = np.random.choice(range(9999), 1)[0]
        self.dataset_name = dataset_name
        self.classes_n = self.classes_n + add_class
        if dataset_name == "moons":
            self.classes_n = 2
            self.X, self.y = make_moons(100, noise=0.1)
            self.X[:, 0] *= 0.5
        elif dataset_name == "blobs":
            self.X, self.y = make_blobs(50 * self.classes_n, centers=self.classes_n, random_state=self.seed)
        elif dataset_name == "spiral":
            spiral_data.seed = self.seed
            self.X, self.y = spiral_data.create_data(100, self.classes_n)
        elif dataset_name == "xor":
            self.classes_n = 2
            self.X = np.c_[
            np.array([
                np.array([np.random.normal(-0.5, 0.1, 25), np.random.normal(0.5, 0.1, 25)]),
                np.array([np.random.normal(-0.5, 0.1, 25), np.random.normal(0.5, 0.1, 25)])
                ]).ravel(),
            np.array([
                np.array([np.random.normal(0.5, 0.1, 25), np.random.normal(-0.5, 0.1, 25)]),
                np.array([np.random.normal(-0.5, 0.1, 25), np.random.normal(0.5, 0.1, 25)])
                ]).ravel()]
            self.y = np.array([0] * 50 + [1] * 50)
        self.X = self.X + np.abs(np.min(self.X))
        self.X = self.X / np.max(self.X)
        self.X = np.clip(0.03, self.X, np.max(self.X) - 0.03)
        self.y_encoded = np.array([[1 if j == i else 0 for j in range(self.classes_n)] for i in self.y])
        global layers
        layers.layers[-1].change_n(self.classes_n)
    
    def to_tensor(self, model_type):
        if model_type == "PyTorch":
            self.X = torch.tensor(self.X).to(torch.float32)
            self.y = torch.tensor(self.y).to(torch.int64)

    def to_array(self, model_type):
        if model_type == "Custom":
            self.X = np.array(self.X.detach())
            self.y = np.array(self.y.detach())


class Layer():
    def __init__(self, n, x_pos, min_n=0, max_n=8, no_buttons=False):
        self.n = n
        self.x_pos = x_pos
        self.min_n = min_n
        self.max_n = max_n
        self.buttons = None
        if not no_buttons:
            button_size = 50
            n_control_button_y_pos = screen_height / 2 + 300
            self.buttons = {"minus": Button(screen, x_pos - button_size - 5, n_control_button_y_pos, button_size, button_size, image=pygame.image.load("sprites/border_50_-.png"), 
                                            onClick=lambda: self.change_n(self.n - 1), inactiveColour=btn_colors[0], hoverColour=btn_colors[1], pressedColour=btn_colors[2]),
                            "add": Button(screen, x_pos + 5, n_control_button_y_pos, button_size, button_size, image=pygame.image.load("sprites/border_50_+.png"), 
                                            onClick=lambda: self.change_n(self.n + 1), inactiveColour=btn_colors[0], hoverColour=btn_colors[1], pressedColour=btn_colors[2]),
                            "activation": Dropdown(screen, x_pos - 35, n_control_button_y_pos + 60, 70, 30, name="ReLU", choices=["ReLU", "Sigmoid"],
                                                   font=pygame.font.SysFont("Calibri", 20), onClick=lambda: stop_fit(),
                                                   inactiveColour=btn_colors[1], hoverColour=btn_colors[2], pressedColour=btn_colors[3])
                             }

    def get_positions(self, distance_between = 70):
        center_y = screen_height / 2
        reverse = lambda a: 0 if a == 1 else 1
        starting_y = center_y - (self.n // 2) * distance_between + distance_between * reverse(self.n % 2) * 0.5 
        positions = []
        for i in range(self.n):
            y = starting_y + i * distance_between
            positions.append((self.x_pos, y))
        self.positions = positions
        return positions
    
    def draw_layer(self, radius=30, width=3):
        for pos in self.positions:
            x, y = pos
            pygame.draw.circle(screen, "white", (x, y), radius)
            pygame.draw.circle(screen, "black", (x, y), radius, width)

    def change_n(self, a):
        stop_fit()
        self.n = sorted((self.min_n, a, self.max_n))[1]


class Layers():
    def __init__(self):
        self.start_x = 100
        self.end_x = 700
        self.layers = [Layer(2, self.start_x, 2, 2, no_buttons=True), 
                       Layer(4, 300), 
                       Layer(4, 500), 
                       Layer(2, self.end_x, no_buttons=True)]
        
    def update_positions(self):
        hidden_layers_n = len(self.layers) - 2
        distance_between = (self.end_x - self.start_x) / (hidden_layers_n + 1)
        x_pos = self.start_x + distance_between
        for i in range(1, len(self.layers) - 1):
            keys = list(self.layers[i].buttons.keys())
            for key in keys:
                self.layers[i].buttons[key].hide()
                del self.layers[i].buttons[key]
            self.layers[i] = Layer(self.layers[i].n, x_pos)
            x_pos += distance_between

    def delete_empty_layers(self):
        removed = False
        for layer in self.layers:
            if layer.n == 0:
                removed = True
                keys = list(layer.buttons.keys())
                for key in keys:
                    layer.buttons[key].hide()
                    del layer.buttons[key]
                self.layers.remove(layer)
        if removed:
            self.update_positions()

    def add_layer(self):
        stop_fit()
        if len(self.layers) - 2 > 2:
            return
        self.layers.append(self.layers[-1])
        self.layers[-2] = Layer(4, 0)
        self.update_positions()


layers = Layers()
dataset = Dataset(model_type)

fig_width = 380
fig_pos = (800, screen_height / 2 - fig_width / 2)

button_add_layer = Button(screen, 360, 70, 80, 80, image=pygame.image.load("sprites/border_80_+.png"), onClick=lambda: layers.add_layer(),
                          inactiveColour=btn_colors[0], hoverColour=btn_colors[1], pressedColour=btn_colors[2])  
button_minus_class = Button(screen, fig_pos[0] + fig_width + 20, screen_height / 2 - 55, 50, 50, image=pygame.image.load("sprites/border_50_-.png"), 
                            onClick=lambda: dataset.load_data(dataset.dataset_name, add_class=-1),
                            inactiveColour=btn_colors[0], hoverColour=btn_colors[1], pressedColour=btn_colors[2])
button_add_class = Button(screen, fig_pos[0] + fig_width + 20, screen_height / 2 + 5, 50, 50, image=pygame.image.load("sprites/border_50_+.png"), 
                          onClick=lambda: dataset.load_data(dataset.dataset_name, add_class=1),
                          inactiveColour=btn_colors[0], hoverColour=btn_colors[1], pressedColour=btn_colors[2])

button_fit = Button(screen, fig_pos[0] + 100, 70, 80, 80, image=pygame.image.load("sprites/border_80_fit.png"), onClick=lambda: start_fit(layers),
                          inactiveColour=btn_colors[0], hoverColour=btn_colors[1], pressedColour=btn_colors[2]) 
button_stop = Button(screen, fig_pos[0] + 200, 70, 80, 80, image=pygame.image.load("sprites/border_80_stop.png"), onClick=lambda: stop_fit(),
                          inactiveColour=btn_colors[0], hoverColour=btn_colors[1], pressedColour=btn_colors[2]) 

button_data_moons = Button(screen, fig_pos[0], fig_pos[1] - 100, 80, 80, image=pygame.image.load("sprites/moons.png"), onClick=lambda: dataset.load_data("moons"),
                           inactiveColour=btn_colors[0], hoverColour=btn_colors[1], pressedColour=btn_colors[2])
button_data_blobs = Button(screen, fig_pos[0] + 100, fig_pos[1] - 100, 80, 80, image=pygame.image.load("sprites/blobs.png"), onClick=lambda: dataset.load_data("blobs"),
                           inactiveColour=btn_colors[0], hoverColour=btn_colors[1], pressedColour=btn_colors[2])
button_data_xor = Button(screen, fig_pos[0] + 200, fig_pos[1] - 100, 80, 80, image=pygame.image.load("sprites/xor.png"), onClick=lambda: dataset.load_data("xor"),
                         inactiveColour=btn_colors[0], hoverColour=btn_colors[1], pressedColour=btn_colors[2])
button_data_spiral = Button(screen, fig_pos[0] + 300, fig_pos[1] - 100, 80, 80, image=pygame.image.load("sprites/spiral.png"), onClick=lambda: dataset.load_data("spiral"),
                            inactiveColour=btn_colors[0], hoverColour=btn_colors[1], pressedColour=btn_colors[2])

dropdown_lr = Dropdown(screen, 105, 10, 75, 30, name="0.01", choices=["3", "1", "0.3", "0.1", "0.03", "0.01", "0.003", "0.001", "0.0003", "0.0001", "0.00003", "0.00001"],
                       font=pygame.font.SysFont("Calibri", 20),
                       inactiveColour=btn_colors[1], hoverColour=btn_colors[2], pressedColour=btn_colors[3])
dropdown_bs = Dropdown(screen, 205, 10, 75, 30, name="10", choices=["1", "10", "50", "100", "All"],
                       font=pygame.font.SysFont("Calibri", 20),
                       inactiveColour=btn_colors[1], hoverColour=btn_colors[2], pressedColour=btn_colors[3])
dropdown_backend = Dropdown(screen, 5, 10, 75, 30, name="Custom", choices=["Custom", "PyTorch"],
                       font=pygame.font.SysFont("Calibri", 20), onClick=lambda: stop_fit(),
                       inactiveColour=btn_colors[1], hoverColour=btn_colors[2], pressedColour=btn_colors[3])




model = nn.Model(layers)
optimizer = None
#model.fit(dataset.X, dataset.y_encoded, epochs=10000, learning_rate=0.1, loss_fn=nn.MSE(), batch_size=10)
fit_model = False


def set_learning_rate():
    if dropdown_lr.getSelected() == None:
        return 0.01
    return float(dropdown_lr.getSelected())

def set_batch_size():
    if dropdown_bs.getSelected() == None:
        return 10
    elif dropdown_bs.getSelected() == "All":
        return len(dataset.X)
    return int(dropdown_bs.getSelected())

learning_rate = set_learning_rate()
batch_size = set_batch_size()

total_epochs = 0
while running:
    events = pygame.event.get() 
    for event in events:
        if event.type == pygame.QUIT:
             running = False

    learning_rate = set_learning_rate()
    batch_size = set_batch_size()

    if fit_model:
        if model_type == "PyTorch":
            epochs = 100
            model.fit(dataset.X, dataset.y, epochs=epochs, loss_fn=torch_nn.CrossEntropyLoss(), batch_size=batch_size, optimizer=optimizer)
        else:
            epochs = 10
            model.fit(dataset.X, dataset.y, epochs=epochs, learning_rate=learning_rate, loss_fn=nn.CategoricallCrossEntropy(), batch_size=batch_size)
        total_epochs += epochs

    #Drawing
    screen.fill("white")

    screen.blit(pygame.font.SysFont("Calibri", 50).render(str(total_epochs), True, "black"), (500, 100))
    
    plot_prediction(fig_pos, fig_width, model, ((255,204,204), (204,238,255), (221,255,204), (255,255,204)))
    plot(fig_pos, fig_width, dataset.X, dataset.y, ("red", "blue", "green", "gold"), model_type)

    positions = []
    for layer in layers.layers:
        poss = layer.get_positions()
        if len(poss) > 0:
            positions.append(poss)
    draw_weights(positions)
    for layer in layers.layers:
        layer.draw_layer()
    layers.delete_empty_layers()

    pygame_widgets.update(events) 
    pygame.display.flip()
    clock.tick(60)

pygame.quit()