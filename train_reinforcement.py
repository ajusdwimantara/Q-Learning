import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import numpy as np
import random


environment_rows = 13*8
environment_columns = 82

q_values = np.zeros((environment_rows, environment_columns, 2))

actions = ['right', 'left']

rewards = np.full((environment_rows, environment_columns), -1)

#assign the reward
for i in range(0, environment_columns):
    for j in range(0, 2):
        rewards[8*8-j][i] = 100 - (j*5)
        rewards[8*8+j][i] = 100 - (j*5)
for i in range(0, environment_columns):
    for j in range(2, 4):
        rewards[8*8-j][i] = 20 + (j*3)
        rewards[8*8+j][i] = 40 + (j*3)
for i in range(0, environment_columns):
    for j in range(4, 6):
        rewards[8*8-j][i] = 5 + (j*2)
        rewards[8*8+j][i] = 20 + (j*2)
for i in range(0, environment_columns):
   for j in range(47, 0, -1):
      rewards[j][i] = -10 - ((48-j)*1)
for i in range(3, environment_columns):
    for j in range(0, 54):
        rewards[j][i] -= 5

#assign the terminal state
for i in range(0, environment_rows):
    rewards[i][81] = -100

for i in range(0, environment_columns):
   rewards[environment_rows-1][i] = -100
   rewards[0][i] = -100

print(rewards)

## Helper function ##

def is_terminal_state(current_row_index, current_column_index):
    if rewards[current_row_index, current_column_index] == -100:
        return True
    else:
        return False

#define a function that will choose a random, non-terminal starting location
def get_starting_location():
  #get a random row and column index
  current_row_index = np.random.randint(environment_rows)
  current_column_index = np.random.randint(environment_columns)
  #continue choosing random row and column indexes until a non-terminal state is identified
  while is_terminal_state(current_row_index, current_column_index):
    current_row_index = np.random.randint(environment_rows)
    current_column_index = np.random.randint(environment_columns)
  return current_row_index, current_column_index

#define an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)
def get_next_action(current_row_index, current_column_index, epsilon):
  #if a randomly chosen value between 0 and 1 is less than epsilon,
  #then choose the most promising value from the Q-table for this state.
  if np.random.random() < epsilon:
    return np.argmax(q_values[current_row_index, current_column_index])
  else: #choose a random action
    return np.random.randint(2)
  
def get_next_location(current_row_index, current_column_index, action_index):
  new_row_index = current_row_index
  new_column_index = current_column_index
  #cube falling
  new_column_index += 1
  #train action
  if actions[action_index] == 'right' and current_row_index < environment_rows - 1:
    new_row_index += 1
  elif actions[action_index] == 'left' and current_row_index > 0:
    new_row_index -= 1
  return new_row_index, new_column_index

def get_shortest_path(start_row_index, start_column_index):
  #return immediately if this is an invalid starting location
  if is_terminal_state(start_row_index, start_column_index):
    return []
  else: #if this is a 'legal' starting location
    current_row_index, current_column_index = start_row_index, start_column_index
    shortest_path = []
    action_path = []
    shortest_path.append([current_row_index, current_column_index])
    #continue moving along the path until we reach the goal (i.e., the item packaging location)
    while not is_terminal_state(current_row_index, current_column_index):
      #get the best action to take
      action_index = get_next_action(current_row_index, current_column_index, 1.)
      action_path.append(action_index)
      #move to the next location on the path, and add the new location to the list
      current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
      shortest_path.append([current_row_index, current_column_index])
    return shortest_path, action_path

### Train ###
#define training parameters
epsilon = 0.8 #the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.6 #discount factor for future rewards
learning_rate = 0.5 #the rate at which the AI agent should learn

#train until it reached the reward
while(np.max(q_values)< 200):
    for episode in range(2000):
        #get the starting location for this episode
        row_index, column_index = get_starting_location()

        #continue taking actions (i.e., moving) until we reach a terminal state
        #(i.e., until we reach the item packaging area or crash into an item storage location)
        while not is_terminal_state(row_index, column_index):
            #choose which action to take (i.e., where to move next)
            action_index = get_next_action(row_index, column_index, epsilon)

            #perform the chosen action, and transition to the next state (i.e., move to the next location)
            old_row_index, old_column_index = row_index, column_index #store the old row and column indexes
            row_index, column_index = get_next_location(row_index, column_index, action_index)

            #receive the reward for moving to the new state, and calculate the temporal difference
            reward = rewards[row_index, column_index]
            old_q_value = q_values[old_row_index, old_column_index, action_index]
            temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value

            #update the Q-value for the previous state and action pair
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            q_values[old_row_index, old_column_index, action_index] = new_q_value

print('Training complete!')
print(np.max(q_values))

#getting the shortest path and the action it takes from the initial position
shortest_path, action_path = get_shortest_path(48, 0)
print(shortest_path)

##########################################################

#initialize input values
trials=2
incl_angle = 0 #no incline angle
g=10
mass_cart=100 # [kg]

trials_global=trials

dt=0.0625
t0=0
t_end=5
t=np.arange(t0,t_end+dt,dt)

F_g=-mass_cart*g

displ_rail=np.zeros((trials,len(t)))
v_rail=np.zeros((trials,len(t)))
a_rail=np.zeros((trials,len(t)))
pos_x_train=np.zeros((trials,len(t)))
pos_y_train=np.zeros((trials,len(t)))
e=np.zeros((trials,len(t)))
e_dot=np.zeros((trials,len(t)))
e_int=np.zeros((trials,len(t)))

pos_x_cube=np.zeros((trials,len(t)))
pos_y_cube=np.zeros((trials,len(t)))

F_ga_t=F_g*np.sin(incl_angle) #tangential component of the gravity force
init_pos_x=60
init_pos_y=120*np.tan(incl_angle)+6.5
init_displ_rail= (init_pos_x**2+init_pos_y**2)**(0.5)
init_vel_rail=0
init_a_rail=0

init_pos_x_global=120 #used for determining the dimensions of the animation window.

trials_magn=trials
history=np.ones(trials)
while(trials>0): #determines how many times cube falls down
    pos_x_cube_ref = 80
    pos_y_cube_ref = 60
    times=trials_magn-trials
    pos_x_cube[times]=pos_x_cube_ref
    pos_y_cube[times]=pos_y_cube_ref-g/2*t**2
    win=False
    delta=1
    F_net = 0

    for i in range(1,len(t)):
        #insert the initial values into the beginning of the predefined arrays.
        if i==1:
            displ_rail[times][0]=init_displ_rail
            pos_x_train[times][0]=init_pos_x
            pos_y_train[times][0]=init_pos_y
            v_rail[times][0]=init_vel_rail
            a_rail[times][0]=init_a_rail

        # 1540 is the approximate force needed to move the train 1 grid/second
        if action_path[i-1] == 0:
           F_net = 1540
        elif action_path[i-1] == 1:
           F_net = -1540
        
        displ_rail[times][i]=displ_rail[times][i-1]+(F_net/mass_cart)*dt
        pos_x_train[times][i]=displ_rail[times][i]*np.cos(incl_angle)
        pos_y_train[times][i]= init_pos_y

        pos_x_cube[times][i]=pos_x_cube[times][i]
        pos_y_cube[times][i]=pos_y_cube[times][i] - 1/2*g*(dt**2)

        #try to catch it
        if (pos_x_train[times][i]-5<pos_x_cube[times][i]+3 and pos_x_train[times][i]+5>pos_x_cube[times][i]-3) or win==True:
            if (pos_y_train[times][i]<pos_y_cube[times][i]-2 and pos_y_train[times][i]+8>pos_y_cube[times][i]) or win==True:
                win=True
                if delta==1:
                    change=pos_x_train[times][i]-pos_x_cube[times][i]
                    delta=0
                pos_x_cube[times][i]=pos_x_train[times][i]-change+0.4
                pos_y_cube[times][i]=pos_y_train[times][i]+5

    init_pose_x = 60
    init_pos_y = 120*np.tan(incl_angle)+6.5
    init_displ_rail= (init_pos_x**2+init_pos_y**2)**(0.5)
    init_vel_rail=v_rail[times][-1]
    init_a_rail=a_rail[times][-1]
    history[times]=delta
    trials=trials-1

############################## ANIMATION #################################
len_t=len(t)
frame_amount=len(t)*trials_global
def update_plot(num):

    platform.set_data([pos_x_train[int(num/len_t)][num-int(num/len_t)*len_t]-3.1,\
    pos_x_train[int(num/len_t)][num-int(num/len_t)*len_t]+3.1],\
    [pos_y_train[int(num/len_t)][num-int(num/len_t)*len_t],\
    pos_y_train[int(num/len_t)][num-int(num/len_t)*len_t]])

    cube.set_data([pos_x_cube[int(num/len_t)][num-int(num/len_t)*len_t]-1,\
    pos_x_cube[int(num/len_t)][num-int(num/len_t)*len_t]+1],\
    [pos_y_cube[int(num/len_t)][num-int(num/len_t)*len_t],\
    pos_y_cube[int(num/len_t)][num-int(num/len_t)*len_t]])

    if trials_magn*len_t==num+1 and num>0: # All attempts must be successful
        if sum(history)==0:
            success.set_text('CONGRATS!')
        else:
            again.set_text('TRY AGAIN')

    return platform,cube,success,again

fig=plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))
gs=gridspec.GridSpec(4,3)

# Create main window
ax_main=fig.add_subplot(gs[0:3,0:2],facecolor=(0.9,0.9,0.9))
plt.xlim(0,init_pos_x_global)
plt.ylim(0,init_pos_x_global)
plt.xticks(np.arange(0,init_pos_x_global+1,10))
plt.yticks(np.arange(0,init_pos_x_global+1,10))
plt.grid(True)

rail=ax_main.plot([0,init_pos_x_global],[5,init_pos_x_global*np.tan(incl_angle)+5],'k',linewidth=6)
platform,=ax_main.plot([],[],'b',linewidth=18)
cube,=ax_main.plot([],[],'k',linewidth=14)

bbox_props_success=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='g',lw='1')
success=ax_main.text(40,60,'',size='20',color='g',bbox=bbox_props_success)

bbox_props_again=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='r',lw='1')
again=ax_main.text(30,60,'',size='20',color='r',bbox=bbox_props_again)

pid_ani=animation.FuncAnimation(fig,update_plot,
    frames=frame_amount,interval=20,repeat=False,blit=True)
plt.show()

























##################################
