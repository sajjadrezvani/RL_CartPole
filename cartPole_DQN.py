# import keras 
import numpy as np 
import random 
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from collections import deque
from tensorflow import gather_nd 
from tensorflow.keras.losses import mean_squared_error
import keras 
from keras.utils import custom_object_scope
# import gymnasium as gym
import gym


class DQL():
    def __init__(self, env, gamma, epsilon, num_episodes):
        self.env = env 
        self.gamma = gamma
        self.epsilon= epsilon
        self.num_episodes= num_episodes

        # dimension: 
        self.state_dimension = 4
        # actions
        self.action_dimension = 2 
        # max size of replay buffer
        self.replay_buffer_size = 300
        # size of training batch sampled from replay buffer
        self.batch_replay_buffer_size = 100
        # update tershold and update counter 
        self.update_target_network_tershold= 100
        self.counter_update_target_network= 0
        
        self.sum_rewards_episode= []
        self.replay_buffer = deque( maxlen=self.replay_buffer_size)

        self.main_network = self.creat_network()
        
        self.target_network = self.creat_network()

        self.target_network.set_weights( self.main_network.get_weights() )


        # this list is used in the cost function to select certain entries of the 
        # predicted and true sample matrices in order to form the loss
        self.action_list =[]    


    def creat_network(self):
        model = Sequential()
        model.add( Dense(128, input_dim=self.state_dimension, activation= 'relu') )
        model.add( Dense(56, activation= 'relu') )
        model.add( Dense(self.action_dimension, activation= 'linear') )

        model.compile( optimizer=RMSprop(), loss = self.custom_loss, metrics=['accuracy'] )
        
        return model
    

    # the selection is performed on the basis of the action indices in the list  self.actionsAppend
    def custom_loss( self, y_true, y_pred):
        s1,s2 = y_true.shape

        # size = self.batch_replay_buffer_size , 2
        indices= np.zeros( shape=(s1,s2))
        indices[:,0]= np.arange(s1)
        indices[:,1]= self.action_list

        # extract slices from a tensor by specifying the indices of the elements you want to retrieve.
        loss = mean_squared_error( gather_nd(y_true, indices=indices.astype(int)) /
                                   gather_nd(y_pred, indices=indices.astype(int)) )
        return loss

    #main training function
    def training_episodes(self):
        
        for index_episode in range(self.num_episodes):
            
            # list of error per episode 
            reward_episode = []

            print(f'\n>>>learning episode {index_episode}th /{self.num_episodes}:')
            
            current_state = self.env.reset()[:4]


            terminal_state = False
            
            num = 0
            while not terminal_state:
                
                num += 1

                env.render()
                action = self.selectAction( current_state, index_episode)
                print( f'\n $$$ In the {num-1}th iteration of {index_episode}th learning episode before reaching terminal the action is {action}!')
                print('action>>>>>>>>>>>>>>>> ' ,action)

                # print(self.env.step(action))
                (next_state, reward, terminal_stat , _ , _) = self.env.step(action)
                reward_episode.append(reward)
                self.replay_buffer.append( (current_state, action, reward, next_state, terminal_state))

                self.train_network()

            
        return 



    def selectAction(self, state, index):
        
        #firstly just explore
        if index < 1: 
            return np.random.choice( self.action_dimension) 
        
        random_sample = np.random.random()

        # move from explore to exploit 
        if index>200: 
            self.epsilon = 0.999*self.epsilon

        # exploration
        if random_sample < self.epsilon:
            return np.random.choice( self.action_dimension)
        
        #exploitation
        else: 
            qValues = self.main_network.predict( state.reshape(1,4) )
            #choose randomly between all max values of qValues
            return np.random.choice( np.where(qValues[0,:]==np.max(qValues[0,:]))[0] )
        


    def train_network(self):
        
        current_state_batch = np.zeros( shape=(self.batch_replay_buffer_size ,4))
        next_state_batch = np.zeros( shape=(self.batch_replay_buffer_size, 4))
        
        # is replay buffer full?
        if( len(self.replay_buffer)> self.batch_replay_buffer_size) :
            print(f'\n*** Now repaly buffer is full {len(self.replay_buffer)} complete')

            #sample a batch from replay buffer 
            random_sample_batch = random.sample(self.replay_buffer , self.batch_replay_buffer_size)

            #enumerate the tuple entries of the randomSampleBatch
            for index, tuples in enumerate(random_sample_batch):
                current_state_batch[index, :] = tuples[0]
                next_state_batch[index, :] = tuples[3]

            Qnext_target_network = self.target_network.predict(next_state_batch)
            Qcurrent_main_network = self.main_network.predict(current_state_batch)

            # in & out network for training 
            input_network = current_state_batch
            output_network = np.zeros( shape=(self.batch_replay_buffer_size,2 ))

            self.action_list= []

            for index, (current_state, action, reward, next_state, teminated) in enumerate(random_sample_batch):
                print( f'--- learning interaion {index}th /{self.batch_replay_buffer_size} in a batch')
                if teminated:
                    y = reward
                else:         # Bellman equation 
                    y = reward + self.gamma*np.max(Qnext_target_network[index])
                
                self.action_list.append(action)

                output_network[index] = Qcurrent_main_network[index]
                output_network[index, action] = y
            

            
            self.main_network.fit( input_network, output_network, batch_size= self.batch_replay_buffer_size, verbose=0, epochs=0)
            
            # target net counter plus
            self.counter_update_target_network+=1
            print(f'\n>>>counter_update_target_network {self.counter_update_target_network}th /{self.update_target_network_tershold} to update target')

            #update target network 
            if (self.counter_update_target_network >= self.update_target_network_tershold):
                
                self.target_network.set_weights( self.main_network.get_weights() )
                print( "target updated --- counter value: " , self.counter_update_target_network,'\n')
                #target net counter reset 
                self.counter_update_target_network =0 


###########################################################################################

env= gym.make( 'CartPole-v1')   #render_mode= "rgb_array"

env.reset()

gamma = 1
# epsilon greedy
epsilon = 0.1
num_episodes = 1000 #1000

deepQ_object = DQL( env, gamma, epsilon, num_episodes)

deepQ_object.training_episodes()

print( deepQ_object.sum_rewards_episode )

print( deepQ_object.main_network.summary() )

deepQ_object.main_network.save('deepQ_network_model.h5')



# import keras2onnx
# ########## convert model to ONNX
# onnx_model = keras2onnx.convert_keras(deepQ_object,         # keras model
#                          name="example",           # the converted ONNX model internal name                     
#                          target_opset=9,           # the ONNX version to export the model to
#                          channel_first_inputs=None # which inputs to transpose from NHWC to NCHW
#                          )
# onnx.save_model(onnx_model, "example.onnx")


with custom_object_scope({'custom_loss': DQL.custom_loss}):
    loaded_model = keras.models.load_model('deepQ_network_model.h5')

    

sum_rewards = 0 

env = gym.make('CartPole-v1')

currentState = env.reset()[:4]
print('$$$$$$$$$$$$$$$$$current: ' ,currentState)
video_length = 400
env = gym.wrappers.RecordVideo(env , 'stored_video', video_length= video_length)


terminal_state = False

while not terminal_state:
    Qvalues = loaded_model.predict(currentState.reshape(1,4))

    env.render()
    action = np.random.choice( np.where(Qvalues[0,:] == np.max(Qvalues[0,:]))[0] )
    print('action###################' , action)
    (currentState , currentReward, terminalState, _, _) = env.step(action)
    sum_rewards += currentReward

env.reset()
env.close()