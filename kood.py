import pandas as pd
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pygame

#1970 - 2001 normal
#2002 - 2017 split two
#depth from -5 to 736
#mag from 2 to 9
#long from -179 to 180
#lat from -84 to 87
#long 1
#lat 135

#COLOURS
green = (0,255,0)
blue = (0,0,255)
red = (255,0,0)
grey = (127,127,127)
black = (0,0,0)
white = (255,255,255)
yellow = (255,255,0)

min_depth = -5
max_depth = 736
first_year = 2004
last_year = 2016
first_year_where_split = 2004 #actual 2002
min_mag = 2
max_mag = 9
min_long = -179
max_long = 180
min_lat = -84
max_lat = 87

long_len = max_long - min_long + 1
lat_len = max_lat - min_lat + 1
mag_len = max_mag - min_mag + 1

square_size = 3
bar_height = 100
border_thickness = 10
control_width = 50
control_display_width = 100
text_size = 40
border_color = blue
background_color = white
button_text = black
button_background = yellow
x_length = long_len * square_size
y_length = lat_len * square_size + bar_height + border_thickness
screen_size = (x_length, y_length)
screen_name = "Display"
game_exit = False
game_speed = 30
game_display = None
clock = None
day = 0
mag = 0
model = None
iterations = 1000
final_reduction = 10
learning_rate = 0.0001
hidden_node_amount = 100

decrease_time_button = None
increase_time_button = None
decrease_mag_button = None
increase_mag_button = None

data_names = ["type", "magType", "mag", "depth", "longitude", "latitude", "time"]

class Button():
    def __init__(self, text, colour, back_ground_colour, x, y, x_length, y_length, text_size):
        self.text_str = text
        self.colour = colour
        self.x = x
        self.y = y
        self.x_length = x_length
        self.y_length = y_length
        self.font = pygame.font.SysFont(None, text_size)
        self.text = self.font.render(self.text_str, True, self.colour)
        self.back_ground_colour = back_ground_colour

    def blit_button(self):
        game_display.blit(self.text, [self.x, self.y])

    def draw_back_ground(self):
        game_display.fill(self.back_ground_colour, rect=[self.x, self.y, self.x_length, self.y_length])

    def draw_button(self):
        self.draw_back_ground()
        self.blit_button()

    def in_button(self, coordinates):
        if coordinates[0] >= self.x and coordinates[0] <= self.x + self.x_length and coordinates[1] >= self.y and coordinates[1] <= self.y + self.y_length:
            return True
        return False

def generate_names(start, first_split, end):
    file_list = []
    for i in range(start, first_split):
        file_name = str(i) + ".csv"
        file_list.append(file_name)
    for i in range(first_split, end+1):
        for j in range(1,3):
            file_name = str(i) + "_" + str(j) + ".csv"
            file_list.append(file_name)
    return file_list

def remove_info(file_names, columns_to_keep):
    for file_name in file_names:
        file = pd.read_csv(file_name)
        
        new_file = file[columns_to_keep]
        new_file.to_csv(file_name)

def min_max_data(file_names):
    values = [[None, None], [None, None], [None, None], [None, None]]
    value_names = ["mag", "longitude", "latitude"]
    for file_name in file_names:
        file = pd.read_csv(file_name)
        for i,value_name in enumerate(value_names):
            if values[i][0] == None or values[i][0] > file[value_name].min():
                values[i][0] = file[value_name].min()
            if values[i][1] == None or values[i][1] < file[value_name].max():
                values[i][1] = file[value_name].max()
    print(values)

def get_distribution(data_type, min_value, max_value, file_names):
    values = [0] * (max_value - min_value)
    for file_name in file_names:
        file = pd.read_csv(file_name)
        for row in file.itertuples():
            depth = getattr(row, data_type)
            if not math.isnan(depth):
                index = int(depth - min_value)
                values[index] += 1
    plt.plot(values)
    plt.show()

def change_stats_to_float(file_names):
    for file_name in file_names:
        file = pd.read_csv(file_name)

        new_file = file[["mag","longitude","latitude"]].fillna(0.0).astype(int)
        new_file["time"] = file["time"]
        new_file.to_csv(file_name)

def change_time(file_names):
    for file_name in file_names:
        file = pd.read_csv(file_name)

        new_file = file[["mag","longitude","latitude"]]
        new_file["time"] = file["time"]
        for i in range(len(file["time"])):
            new_file.at[i, "time"] = str(file.loc[i]["time"])[:10]
        #new_file["time"] = file["time"].str[:10]
        new_file.to_csv(file_name)

def reverse(file_names):
    for file_name in file_names:
        file = pd.read_csv(file_name)

        new_file = file.iloc[::-1]
        new_file.to_csv(file_name)

def max_location_of_quakes(mag_to_find):
    file = pd.read_csv(str(mag_to_find) + ".csv")
    return(file.max)

def read_file_into_array(array, file_name):
    file = pd.read_csv(file_name)
    last_date = None
    numpy_array = None
    for i in range(len(file)):
        if file.loc[i]["time"] != last_date:
            last_date = file.loc[i]["time"]
            if numpy_array is not None:
                array.append(numpy_array)
            numpy_array = np.zeros((long_len, lat_len, mag_len), dtype = int)
            
        long = file.loc[i]["longitude"] - min_long
        lat = file.loc[i]["latitude"] - min_lat
        mag = file.loc[i]["mag"] - min_mag
        numpy_array[long][lat][mag] = 1
    array.append(numpy_array)

def read_file_into_earthquake_map(quake_map, file_name):
    file = pd.read_csv(file_name)
    for i in range(len(file)):
        this_data = file.loc[i]
        quake_map[this_data[1] - min_mag][this_data[2] - min_long][this_data[3] - min_lat] += 1

def save_map(quake_map):
    for i, mag_map in enumerate(quake_map):
        quake_df = pd.DataFrame(data = mag_map)
        quake_df.to_csv(str(i + min_mag) + ".csv")

def make_general_earthquake_map(file_names):
    earthquake_map = np.zeros((mag_len, long_len, lat_len), dtype = int)
    for file_name in file_names:
        read_file_into_earthquake_map(earthquake_map, file_name)
        print(file_name)
    print("map done")
    save_map(earthquake_map)

def make_data_table(file_names):
    array_of_numpys = []
    for file_name in file_names:
        read_file_into_array(array_of_numpys, file_name)
        print(file_name)
    print("table done")
    print(len(array_of_numpys))
    return array_of_numpys

def make_3x3_vector_direct(file_names, loc):
    vector_list = []
    answer_list = []

    possible_locs = gen_possible_locs(loc)

    for file_name in file_names:
        file = pd.read_csv(file_name)
        last_date = None
        one_vector = None
        one_answer = None
        sum_of_quakes = 0
        for i in range(len(file)):
            this_data = file.loc[i]
            if this_data[4] != last_date:
                last_date = this_data[4]
                if one_vector is not None:
                    vector_list.append(one_vector)
                    answer_list.append(one_answer)
                one_vector = np.zeros((9 * mag_len), dtype = int)
                one_answer = np.zeros((mag_len), dtype = int)

            long = this_data[2] - min_long
            lat = this_data[3] - min_lat
            for j, loca in enumerate(possible_locs):
                if long == loca[0] and lat == loca[1]:
                    vector_location = j * mag_len + this_data[1] - min_mag
                    one_vector[vector_location] = 1
                    sum_of_quakes += 1
                    if j == 4:
                        one_answer[this_data[1]] = 1
        vector_list.append(one_vector)
        answer_list.append(one_answer)
        if sum_of_quakes > 0:
            print(file_name)
            print(sum_of_quakes)
    print("vectors done")
    del answer_list[0]
    return vector_list, answer_list

def gen_possible_locs(loc):
    loc_list = []
    
    left_loc = loc[0] - 1
    right_loc = loc[0] + 1
    if left_loc < 0:
        left_loc = 359
    if right_loc > 359:
        right_loc = 0

    for i in range(3):
        loc_list.append([left_loc, loc[1] - 1 + i])
        loc_list.append([loc[0], loc[1] - 1 + i])
        loc_list.append([right_loc, loc[1] - 1 + i])

    return loc_list

def make_3x3_vector_from_data_table(data_table, loc):
    vector_list = []
    
    left_loc = loc[0] - 1
    right_loc = loc[0] + 1
    if left_loc < 0:
        left_loc = 359
    if right_loc > 359:
        right_loc = 0
    
    for k, day in enumerate(data_table):
        #one vector will become a 1-d vector with length 9*mag_values starting from top left going right and then down
        one_vector = []
        for i in range(3):
            for j in range(len(day[left_loc][loc[1] - 1 + i])):
                one_vector.append(day[left_loc][loc[1] - 1 + i][j])
            for j in range(len(day[loc[0]][loc[1] - 1 + i])):
                one_vector.append(day[loc[0]][loc[1] - 1 + i][j])
            for j in range(len(day[right_loc][loc[1] - 1 + i])):
                one_vector.append(day[right_loc][loc[1] - 1 + i][j])
        vector_list.append(one_vector)

    #print(vector_list)
    print("vectors done")
    return vector_list

def display(data):
    global game_display
    global clock
    global decrease_time_button
    global increase_time_button
    global decrease_mag_button
    global increase_mag_button
    clock = pygame.time.Clock()
    pygame.font.init()

    #Buttons
    decrease_time_button = Button("-1", button_text, button_background, 0, 0, control_width, bar_height, text_size)
    increase_time_button = Button("+1", button_text, button_background, control_width + control_display_width, 0, control_width, bar_height, text_size)
    decrease_mag_button = Button("-1", button_text, button_background, screen_size[0] - 2*control_width - control_display_width, 0, control_width, bar_height, text_size)
    increase_mag_button = Button("+1", button_text, button_background, screen_size[0] - control_width, 0, control_width, bar_height, text_size)

    game_display = setup()
    game_loop(data)
    pygame.quit()

def mouse_press_event(mouse_event):
    global day
    global mag
    coordinates = list(mouse_event.pos)
    if decrease_time_button.in_button(coordinates):
        day -= 1
    elif increase_time_button.in_button(coordinates):
        day += 1
    elif decrease_mag_button.in_button(coordinates):
        mag -= 1
    elif increase_mag_button.in_button(coordinates):
        mag += 1

def setup():
    pygame.init()
    display = pygame.display.set_mode(screen_size)
    pygame.display.set_caption(screen_name)
    return display

def game_loop(data):
    while not game_exit:
        one_loop(data)

def one_loop(data):
    event_handle()
    draw_game(data)
    clock.tick(game_speed)

def event_handle():
    global game_exit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_exit = True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_press_event(event)

def draw_game(data):
    #Resets the display
    game_display.fill(background_color)

    #Draws the border between control and map
    pygame.draw.rect(game_display, border_color, (0, bar_height, screen_size[0], border_thickness))

    #Draws buttons
    decrease_time_button.draw_button()
    increase_time_button.draw_button()
    decrease_mag_button.draw_button()
    increase_mag_button.draw_button()

    #Draws earthquakes
    for x, long in enumerate(data[day]):
        for y, lat in enumerate(long):
            if lat[mag] == 1:
                game_display.fill(black, rect=[x * square_size, y * square_size + bar_height + border_thickness, square_size, square_size])

    #Actually updates
    pygame.display.update()

def initialize_network(n_input, n_hidden, n_output):
    both_inputs = n_input + n_hidden
    some_model = dict(
        wf = np.random.randn(both_inputs, n_hidden) / np.sqrt(both_inputs / 2.),
        wi = np.random.randn(both_inputs, n_hidden) / np.sqrt(both_inputs / 2.),
        wo = np.random.randn(both_inputs, n_hidden) / np.sqrt(both_inputs / 2.),
        wc = np.random.randn(both_inputs, n_hidden) / np.sqrt(both_inputs / 2.),
        wy = np.random.randn(n_hidden, n_output) / np.sqrt(n_hidden / 2.),
        
        bf = np.ones((1, n_hidden)) / 2,
        bi = np.ones((1, n_hidden)) / 2,
        bo = np.ones((1, n_hidden)) / 2,
        bc = np.ones((1, n_hidden)) / 2,
        by = np.ones((1, n_output)) / 2
        )

    return some_model

def load_network():
    some_model = dict(
        wf = pd.read_csv("wf.csv").as_matrix(),
        wi = pd.read_csv("wi.csv").as_matrix(),
        wo = pd.read_csv("wo.csv").as_matrix(),
        wc = pd.read_csv("wc.csv").as_matrix(),
        wy = pd.read_csv("wy.csv").as_matrix(),

        bf = pd.read_csv("bf.csv").as_matrix(),
        bi = pd.read_csv("bi.csv").as_matrix(),
        bo = pd.read_csv("bo.csv").as_matrix(),
        bc = pd.read_csv("bc.csv").as_matrix(),
        by = pd.read_csv("by.csv").as_matrix()
        )
    for k in some_model.keys():
        some_model[k] = np.delete(some_model[k], 0, axis = 1)
    return some_model

def lstm_forward(X, memory, n_input, n_hidden):
    m = model
    wf, wi, wc, wo, wy = m['wf'], m['wi'], m['wc'], m['wo'], m['wy']
    bf, bi, bc, bo, by = m['bf'], m['bi'], m['bc'], m['bo'], m['by']

    h_old, c_old = memory

    temp_X = X.reshape(1, -1)
    new_X = np.column_stack((h_old, temp_X))

    hf = sigmoid(new_X @ wf + bf)
    hi = sigmoid(new_X @ wi + bi)
    ho = sigmoid(new_X @ wo + bo)
    hc = tanh(new_X @ wc + bc)

    c = hf * c_old + hi * hc
    h = ho * tanh(c)

    y = h @ wy + by
    prob = sigmoid(y)

    state = (h,c)
    cache = (hf, hi, ho, hc, c, h, y, wf, wi, wc, wo, wy, new_X, c_old)

    return prob, state, cache

def lstm_backward(prob, y_train, d_next, cache, n_hidden):
    hf, hi, ho, hc, c, h, y, wf, wi, wc, wo, wy, new_X, c_old = cache
    dh_next, dc_next = d_next

    dy = prob.copy()
    dy -= y_train
    dy = dsigmoid(dy)

    dwy = h.T @ dy
    dby = dy

    dh = dy @ wy.T + dh_next

    dho = tanh(c) * dh
    dho = dsigmoid(ho) * dho

    dc = ho * dh * dtanh(c)
    dc = dc + dc_next

    dhf = c_old * dc
    dhf = dsigmoid(hf) * dhf

    dhi = hc * dc
    dhi = dsigmoid(hi) * dhi

    dhc = hi * dc
    dhc = dtanh(hc) * dhc

    dwf = new_X.T @ dhf
    dbf = dhf
    dXf = dhf @ wf.T

    dwi = new_X.T @ dhi
    dbi = dhi
    dXi = dhi @ wi.T

    dwo = new_X.T @ dho
    dbo = dho
    dXo = dho @ wo.T

    dwc = new_X.T @ dhc
    dbc = dhc
    dXc = dhc @ wc.T

    dX = dXo + dXc + dXi + dXf

    dh_next = dX[:, :n_hidden]

    dc_next = hf * dc

    grad = dict(wf=dwf, wi=dwi, wc=dwc, wo=dwo, wy=dwy, bf=dbf, bi=dbi, bc=dbc, bo=dbo, by=dby)
    state = (dh_next, dc_next)

    return grad, state

def train_step(X_train, y_train, state):
    global model
    probs = []
    caches = []
    #loss = 0.
    loss = []
    for i in range(mag_len):
        loss.append([0,0,0,0])
    h, c = state

    for x, y_true in zip(X_train, y_train):
        prob, state, cache = lstm_forward(x, state, 72, hidden_node_amount)
        #loss += cost_function(prob, y_true.reshape(1, -1))
        true = y_true.copy()
        true = true.reshape(1, -1)
        preds = prob.copy()
        preds = np.rint(preds)
        for i in range(mag_len):
            if true[0][i] == 0 and preds[0][i] == 0:
                loss[i][0] += 1
            elif true[0][i] == 0 and preds[0][i] == 1:
                loss[i][1] += 1
            elif true[0][i] == 1 and preds[0][i] == 0:
                loss[i][2] += 1
            elif true[0][i] == 1 and preds[0][i] == 1:
                loss[i][3] += 1

        probs.append(prob)
        caches.append(cache)

    #loss /= len(X_train)
    for sub in loss:
        for sub_sub in sub:
            sub_sub /= len(X_train)

    d_next = (np.zeros_like(h), np.zeros_like(c))
    grads = {k: np.zeros_like(v) for k, v in model.items()}

    for prob, y_true, cache in reversed(list(zip(probs, y_train, caches))):
        grad, d_next = lstm_backward(prob, y_true, d_next, cache, hidden_node_amount)

        for k in grads.keys():
            grads[k] += grad[k]

    return grads, loss, state

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1.0 - np.tanh(x)**2

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def cross_entropy(p, t):
    return -(1.0) * np.sum(t*np.log(p) + (1-t)*np.log(1-p))

def cost_function(p, t):
    return np.sum((p-t)**2)
    

test_file = ["2017_1.csv"]
warmup_file = ["2016_1.csv", "2016_2.csv"]
valid_file = ["2017_1.csv", "2017_2.csv"]
files = generate_names(first_year, first_year_where_split, last_year)

#data, answers = make_3x3_vector_direct(files, [1, 135])
#model = initialize_network(9 * mag_len, hidden_node_amount, mag_len)
model = load_network()

a = np.ones(hidden_node_amount).reshape(1, -1) / 2
b = np.ones(hidden_node_amount).reshape(1, -1) / 2
init_state = (a,b)


#training
present_state = None
"""for i in range(1000):
    confidence = 1 / (1 + (i / iterations ) * (final_reduction - 1) )
    gradients, loss, present_state = train_step(data, answers, init_state)
    for k in model.keys():
        model[k] -= gradients[k] * learning_rate * confidence
    print(loss)

for k in model.keys():
    df = pd.DataFrame(data = model[k])
    df.to_csv(str(k) + ".csv")"""

#warmup for test
data, answers = make_3x3_vector_direct(warmup_file, [1, 135])
gradients, loss, present_state = train_step(data, answers, init_state)

print("Test:")

#test
data, answers = make_3x3_vector_direct(valid_file, [1, 135])
gradients, loss, state_something = train_step(data, answers, present_state)
print(loss)

"""for grad in gradients:
    print(grad)
    for i in range(len(gradients[grad])):
        if gradients[grad][i].sum() > 0:
            print(i)"""
#print(gradients["wf"][0])
#print(gradients["wf"][1])
#print(gradients["wf"][74])
#print(gradients["wf"][75])

#print(loss)

print("SUCCESS!")

        


