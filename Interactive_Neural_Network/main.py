import pygame
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
import spiral_data
import numpy as np
from pygame_widgets.button import Button
from pygame_widgets.dropdown import Dropdown
from pygame_widgets.slider import Slider
from pygame_widgets.toggle import Toggle
import pygame_widgets
import neural_network as nn


screen_width = 1280
screen_height = 800

pygame.init()
pygame.display.set_caption("Neuronkė")
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()
running = True
btn_colors = ((255, 255, 255), (230, 230, 230), (205, 205, 205), (180, 180, 180))

def draw_weights(positions, weights):
        for i in range(len(positions) - 1):
            for j in range(len(positions[i])): 
                for k in range(len(positions[i + 1])):
                    if np.isnan(weights[i][j][k]):
                        width = 1
                    else:
                        width = int(weights[i][j][k] + 5)
                    if width <= 0:
                        width = 1
                    pygame.draw.line(screen, "black", positions[i][j], positions[i + 1][k], width)

def plot(fig_top_left, width, data, c, cmap):
    fx, fy = fig_top_left
    pygame.draw.rect(screen, "black", (fx, fy, width, width), 3)
    for d, _c in zip(data, c):
        color = cmap[_c]
        d_0 = d[0]
        d_1 = d[1]
        pygame.draw.circle(screen, color, (fx + d_0 * (width - 8), fy + d_1 * (width - 8)), 4)

def plot_prediction(fig_top_left, width, model, cmap):
    xx, yy = np.meshgrid(np.linspace(fig_top_left[0], fig_top_left[0] - 15 + width, slider_res.getValue()), np.linspace(fig_top_left[1], fig_top_left[1] - 15 + width, slider_res.getValue()))
    X = np.c_[(xx.ravel() - fig_top_left[0]) / width, (yy.ravel() - fig_top_left[1]) / width]
    pred = model.predict(X)
    pred = np.argmax(pred, axis=1)
    for x, y, pr in zip(xx.ravel(), yy.ravel(), pred):
        color = cmap[pr]
        pygame.draw.rect(screen, color, (x, y, 14, 14))

def plot_loss_accuracy(fig_top_left, width, height, loss, accuracy):
    pygame.draw.rect(screen, "black", (fig_top_left, (width, height)), 3)
    if len(loss) != 0:
        loss = np.array(loss)
        loss /= np.max(loss)
        if len(loss) > 1000 and toggle_last_1000.getValue() == True:
            loss = loss[-1000:]
            accuracy = accuracy[-1000:]
        X_l = np.linspace(fig_top_left[0] + 3, fig_top_left[0] - 3 + width, len(loss))
        y_l = fig_top_left[1] + height - loss * (height - 3)
        accuracy = np.array(accuracy)
        X_a = np.linspace(fig_top_left[0] + 3, fig_top_left[0] - 3 + width, len(accuracy))
        y_a = fig_top_left[1] + height - accuracy * (height - 3)
        for i in range(len(X_l) - 1):
            pygame.draw.line(screen, "blue", (X_l[i], y_l[i]), (X_l[i + 1], y_l[i + 1]), 1)
            pygame.draw.line(screen, "darkorange", (X_a[i], y_a[i]), (X_a[i + 1], y_a[i + 1]), 1)
        screen.blit(pygame.font.SysFont("Calibri", 20).render("loss", True, "black"), (fig_top_left[0] + width + 8, y_l[-1] - 6))
        screen.blit(pygame.font.SysFont("Calibri", 20).render("accuracy", True, "black"), (fig_top_left[0] + width + 8, y_a[-1] - 6))



def start_fit(layers):
    global model, fit_model, total_epochs, optimizer, learning_rate
    model = nn.Model(layers, output_activation)
    fit_model = True
    total_epochs = 0

def stop_fit():
    global fit_model
    fit_model = False

def change_activation(layer):
    global fit_model
    fit_model = False
    if layer.buttons["activation"].getSelected() != None:
        layer.activation_name = layer.buttons["activation"].getSelected()


class Dataset():
    def __init__(self):
        self.seed = np.random.choice(range(9999), 1)[0]
        self.dataset_name = "moons"
        self.X, self.y = make_moons(100, noise=0.1)
        self.X[:, 0] *= 0.5
        self.X = self.X + np.abs(np.min(self.X))
        self.X = self.X / np.max(self.X)
        self.X = np.clip(0.03, self.X, np.max(self.X) - 0.03)
        self.classes_n = 2
        self.y_encoded = np.array([[1 if j == i else 0 for j in range(self.classes_n)] for i in self.y])

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


class Layer():
    def __init__(self, n, x_pos, activation_name="Sigmoid", min_n=0, max_n=8, no_buttons=False):
        self.activation_name = activation_name
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
                            "activation": Dropdown(screen, x_pos - 35, n_control_button_y_pos + 60, 70, 30, name=activation_name, choices=["Sigmoid", "ReLU", "Linear"],
                                                   font=pygame.font.SysFont("Calibri", 20), onClick=lambda: stop_fit(),
                                                   inactiveColour=btn_colors[1], hoverColour=btn_colors[2], pressedColour=btn_colors[3], direction="up")
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
    
    def draw_layer(self, biases, radius=30):
        for pos, b in zip(self.positions, biases):
            if np.isnan(b):
                width = 1
            else:
                width = int(b + 5)
            if width <= 0:
                width = 1
            x, y = pos
            pygame.draw.circle(screen, "white", (x, y), radius)
            pygame.draw.circle(screen, "dimgray", (x, y), radius, width)

    def change_n(self, a):
        stop_fit()
        add = False
        if self.n < a:
            add = True
        self.n = sorted((self.min_n, a, self.max_n))[1]
        if add:
            global model, layers
            model = nn.Model(layers, output_activation)


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
            self.layers[i] = Layer(self.layers[i].n, x_pos, self.layers[i].activation_name)
            x_pos += distance_between
        global model
        model = nn.Model(self, output_activation)

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
dataset = Dataset()

fig_width = 380
fig_pos = (800, screen_height / 2 - fig_width / 2)

button_add_layer = Button(screen, 360, fig_pos[1] - 200, 80, 80, image=pygame.image.load("sprites/border_80_+.png"), onClick=lambda: layers.add_layer(),
                          inactiveColour=btn_colors[0], hoverColour=btn_colors[1], pressedColour=btn_colors[2])  
button_minus_class = Button(screen, fig_pos[0] + fig_width + 20, screen_height / 2 - 55, 50, 50, image=pygame.image.load("sprites/border_50_-.png"), 
                            onClick=lambda: dataset.load_data(dataset.dataset_name, add_class=-1),
                            inactiveColour=btn_colors[0], hoverColour=btn_colors[1], pressedColour=btn_colors[2])
button_add_class = Button(screen, fig_pos[0] + fig_width + 20, screen_height / 2 + 5, 50, 50, image=pygame.image.load("sprites/border_50_+.png"), 
                          onClick=lambda: dataset.load_data(dataset.dataset_name, add_class=1),
                          inactiveColour=btn_colors[0], hoverColour=btn_colors[1], pressedColour=btn_colors[2])

button_fit = Button(screen, fig_pos[0] + 100, fig_pos[1] - 200, 80, 80, image=pygame.image.load("sprites/border_80_fit.png"), onClick=lambda: start_fit(layers),
                          inactiveColour=btn_colors[0], hoverColour=btn_colors[1], pressedColour=btn_colors[2]) 
button_stop = Button(screen, fig_pos[0] + 200, fig_pos[1] - 200, 80, 80, image=pygame.image.load("sprites/border_80_stop.png"), onClick=lambda: stop_fit(),
                          inactiveColour=btn_colors[0], hoverColour=btn_colors[1], pressedColour=btn_colors[2]) 

button_data_moons = Button(screen, fig_pos[0], fig_pos[1] - 100, 80, 80, image=pygame.image.load("sprites/moons.png"), onClick=lambda: dataset.load_data("moons"),
                           inactiveColour=btn_colors[0], hoverColour=btn_colors[1], pressedColour=btn_colors[2])
button_data_blobs = Button(screen, fig_pos[0] + 100, fig_pos[1] - 100, 80, 80, image=pygame.image.load("sprites/blobs.png"), onClick=lambda: dataset.load_data("blobs"),
                           inactiveColour=btn_colors[0], hoverColour=btn_colors[1], pressedColour=btn_colors[2])
button_data_xor = Button(screen, fig_pos[0] + 200, fig_pos[1] - 100, 80, 80, image=pygame.image.load("sprites/xor.png"), onClick=lambda: dataset.load_data("xor"),
                         inactiveColour=btn_colors[0], hoverColour=btn_colors[1], pressedColour=btn_colors[2])
button_data_spiral = Button(screen, fig_pos[0] + 300, fig_pos[1] - 100, 80, 80, image=pygame.image.load("sprites/spiral.png"), onClick=lambda: dataset.load_data("spiral"),
                            inactiveColour=btn_colors[0], hoverColour=btn_colors[1], pressedColour=btn_colors[2])

dropdown_lr = Dropdown(screen, 105, fig_pos[1] - 200, 75, 30, name="0.01", choices=["3", "1", "0.3", "0.1", "0.03", "0.01", "0.003", "0.001", "0.0003", "0.0001", "0.00003", "0.00001"],
                       font=pygame.font.SysFont("Calibri", 20),
                       inactiveColour=btn_colors[1], hoverColour=btn_colors[2], pressedColour=btn_colors[3])
dropdown_bs = Dropdown(screen, 205, fig_pos[1] - 200, 75, 30, name="10", choices=["1", "10", "50", "100", "All"],
                       font=pygame.font.SysFont("Calibri", 20),
                       inactiveColour=btn_colors[1], hoverColour=btn_colors[2], pressedColour=btn_colors[3])
dropdown_loss = Dropdown(screen, 50, fig_pos[1] - 160, 300, 30, name="Categorical Cross Entropy", choices=["Categorical Cross Entropy", "MSE"],
                       font=pygame.font.SysFont("Calibri", 20),
                       inactiveColour=btn_colors[1], hoverColour=btn_colors[2], pressedColour=btn_colors[3])

output_activation = Dropdown(screen, 665, screen_height / 2 + 360, 70, 30, name="Softmax", choices=["Softmax", "Sigmoid", "Linear"],
                                                   font=pygame.font.SysFont("Calibri", 20), onClick=lambda: stop_fit(),
                                                   inactiveColour=btn_colors[1], hoverColour=btn_colors[2], pressedColour=btn_colors[3], direction="up")

slider_res = Slider(screen, fig_pos[0], int(fig_pos[1]) - 15, fig_width, 10, min=30, max=200, step=1, color=btn_colors[2])
toggle_last_1000 = Toggle(screen, fig_pos[0], int(fig_pos[1]) + fig_width + 5, 20, 10)


model = nn.Model(layers, output_activation)
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
    screen.fill("white")
    events = pygame.event.get() 
    for event in events:
        if event.type == pygame.QUIT:
             running = False

    learning_rate = set_learning_rate()
    batch_size = set_batch_size()

    epoch_per_second_str = "0 epochs/s"

    '''
    if dropdown_loss.getSelected() == None or dropdown_loss.getSelected() == "Categorical Cross Entropy":
        output_activation.__chosen._value = "Softmax"
    elif dropdown_loss.getSelected() == "MSE" and (output_activation.getSelected() == None or output_activation.getSelected() == "Softmax"):
        output_activation.__chosen._value = "Sigmoid"
        '''

    if fit_model:
        epoch_per_second_str =  f"{str(int(clock.get_fps()))} epochs/s"
        epochs = 1
        if dropdown_loss.getSelected() == None or dropdown_loss.getSelected() == "Categorical Cross Entropy":
            loss_fn = nn.CategoricallCrossEntropy()
            model.fit(dataset.X, dataset.y, epochs=epochs, learning_rate=learning_rate, loss_fn=loss_fn, batch_size=batch_size)
        elif dropdown_loss.getSelected() == "MSE":
            loss_fn = nn.MSE() 
            model.fit(dataset.X, dataset.y_encoded, epochs=epochs, learning_rate=learning_rate, loss_fn=loss_fn, batch_size=batch_size)  
        total_epochs += epochs
    else:
        for layer in layers.layers:
            if layer.buttons != None:
                change_activation(layer)

    #Drawing
    plot_loss_accuracy((fig_pos[0], fig_pos[1] + fig_width + 20), fig_width, 100, model.history["train_loss"], model.history["train_accuracy"])
    screen.blit(pygame.font.SysFont("Calibri", 50).render(str(total_epochs) + " Epochs", True, "black"), (500, fig_pos[1] - 200))
    screen.blit(pygame.font.SysFont("Calibri", 25).render(epoch_per_second_str, True, "black"), (500, fig_pos[1] - 150))
    #screen.blit(pygame.font.SysFont("Calibri", 20).render("display history only from last 1000 epochs", True, "black"), (fig_pos[0] + 20, int(fig_pos[1]) + fig_width + 5))

    plot_prediction(fig_pos, fig_width, model, ((255,204,204), (204,238,255), (221,255,204), (255,255,204)))
    plot(fig_pos, fig_width, dataset.X, dataset.y, ("red", "blue", "green", "gold"))

    positions = []
    for layer in layers.layers:
        poss = layer.get_positions()
        if len(poss) > 0:
            positions.append(poss)
    weights, biases = model.get_weights_and_biases()
    draw_weights(positions, weights)
    for layer, B in zip(layers.layers, biases):
        layer.draw_layer(B)
    layers.delete_empty_layers()

    pygame_widgets.update(events) 
    pygame.display.flip()
    clock.tick()

pygame.quit()
