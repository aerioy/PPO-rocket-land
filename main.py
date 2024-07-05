import pygame, sys, time
import numpy as np
from pygame.locals import QUIT
import random as r
from random import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


torch.set_default_dtype(torch.float64)
def getengineforce(ang,engineang):
    v = np.array([np.cos(ang), np.sin(ang)])
    return [
        np.array(v * np.sin(engineangle) * engineforce),
        (engineforce/(2 * np.pi * 50) * np.cos(engineang))
    ]

gravity = np.array([0,-18])

def newcoordinates(p):
    return np.array([p[0],ydim-p[1]])

def rotate90 (v):
    return np.array([-1 * v[1],v[0]])


def getvertex(pos, ang):
    v = np.array([np.cos(ang), np.sin(ang)])
    w = rotate90(v)
    out = []
    for t in range (-10,10):
        out.append(newcoordinates(pos + v * 70 + (0.2) * (-1 * v) * t**2 + w * t))
    out.append(newcoordinates(pos + (50 * v + 10 * w)))
    out.append(newcoordinates(pos + (-50 * v + 10 * w)))
    out.append(newcoordinates(pos + (-50 * v + (-10) * w)))
    out.append(newcoordinates(pos + (50 * v + (-10)  *w)))
    return out
    
def randrange (a,b):
    return a + (b-a) * random()

dt = 0.023
velocity = np.array([0,0])
pause = False
xdim = 1500
ydim = 1000
fuel = 20
engine = 0
engineangle = np.pi/2
engineforce = 40
groundheight = 100
angle = randrange(np.pi/4,3 * np.pi/4 )
anglevelocity = 0
position = np.array([xdim/2,930])
points = []
state1 = []


"""
state 
0 positoinx 
1 postitiony
2 velocityx
3 velocityy
4 angle
5 anglevelocity
6 engine
7 engineangle
8 fuel

action 

1 -> do nothing
2 -> move engine left
3 -> move engine right
4 -> toggle engine 
"""

def is_terminal(state):
    terminal = False
    for x in getvertex(np.array([state[0],state[1]]),state[4]):
        if x[1] >= 900 or x[1] <= 0 or x[0] >= xdim or x[0] <= 0:
            terminal = True
    return terminal


def reward(state):
    position = np.array([state[0], state[1]])
    target = np.array([xdim/2, 100])
    distance = np.linalg.norm(position - target)
    velocity = np.array([state[2], state[3]])
    speed = np.linalg.norm(velocity)
    angleoffset = abs(np.pi/2 - state[4])**2
    
    reward = -distance / 900  # Encourage getting closer to the target
    reward -= speed / 10  # Penalize high speeds
    reward += state[8] / 20  # Small bonus for conserving fuel
    reward -= angleoffset * 3
    
    if is_terminal(state):
        if position[1] >= 500:
            reward -= 250
        if position[0] <= 0 or position[0] >= xdim:
            reward -= 250
        if position[1] <= 110 and abs(position[0] - xdim/2) < 300:
            reward += 250 -  8 * speed  # Big bonus for landing on the pad
        else:
            reward -= 225  # Big penalty for crashing
    
    return reward

def transition (state,action):
    angle = state[4]
    engine = state[6]
    anglevelocity = state[5]
    engineangle = state[7]
    fuel = state[8]
    if fuel <= 0:
        engine = 0
    if action == 4:
               if engine == 1:
                   engine = 0
               else:
                   if fuel > 0:
                    engine = 1
    if action == 2:
        if engineangle - np.pi/12 >= np.pi/4 and engineangle - np.pi/12 <= 3 *np.pi / 4:
            engineangle -= np.pi/12
    if action == 3:
        if engineangle + np.pi/12 >= np.pi/4 and engineangle + np.pi/12 <= 3 *np.pi / 4:
            engineangle += np.pi/12

    position = np.array([state[0],state[1]])
    velocity = np.array([state[2],state[3]])
    if engine == 1:
        fuel = fuel - dt
    positionout =  position + velocity*dt
    velocityout = velocity +  (gravity + engine * getengineforce(angle,engineangle)[0]) *dt
    angleout = angle + anglevelocity * dt
    anglevelocityout = anglevelocity  + engine * getengineforce(angle,engineangle)[1] * dt
    nextstate = np.array([positionout[0],positionout[1],velocityout[0],velocityout[1],angleout,anglevelocityout,engine,engineangle,fuel])
    return (nextstate)

        
def g(x, y):
    return torch.where(y >= 0, (1+x) * y, (1-x) * y)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=0.01)
        m.bias.data.fill_(0.01)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def flatten2d(arr):
    out = []
    for x in arr:
        for y in x:
            out.append(y)
    return out

class network(nn.Module):
    def __init__(self,layers,lrate,is_distribution = False):
      super(network, self).__init__()
      self.distribution = is_distribution
      self.network = self.constructnet(layers)
      self.optimizer = optim.Adam(self.parameters(),lr = lrate)

    
    def constructnet (self,layers):
      arcitecture = []
      for x in range(len(layers) - 1):
        arcitecture.append(nn.Linear(layers[x],layers[x+1]))
        if x != len(layers) - 2:
         arcitecture.append(nn.ReLU())
      if self.distribution:
        arcitecture.append(nn.Softmax(dim = -1))
      return nn.Sequential(*arcitecture)

    def forward (self,x):
        if self.distribution:
            distribution = self.network(x)
            distribution = Categorical(distribution)
            return distribution
        return self.network(x)
     
    def saveparameters (self,filepath):
        torch.save(self.state_dict(),filepath)
    
    def loadparameters (self,filepath):
        self.load_state_dict(torch.load(filepath))


class memory:
    def __init__ (self):
        self.length = 0
        self.states = []
        self.actions = []
        self.probabilities = []
        self.values = []
        self.rewards = []
        self.trajectories = []
    
    def generatetrajectories(self):
        return self.trajectories
        
    def store (self,state,action,probability,value,reward):
        self.length += 1
        self.states.append(state)
        self.actions.append(action)
        self.probabilities.append(probability)
        self.values.append(value)
        self.rewards.append(reward)
    
    def storetrajectory(self):
        self.trajectories.append([
             np.array(self.states),
             np.array(self.actions),
             np.array(self.probabilities),
             np.array(self.values),
             np.array(self.rewards)
             ])
        self.clear()
    
    def clear(self,clear_trajectories = False):
        self.length = 0
        self.states = []
        self.actions = []
        self.probabilities = []
        self.values = []
        self.rewards = []
        if clear_trajectories:
            self.trajectories = []


class agent:
    def __init__ (self):
        self.policynet = network([9,128,128,4],0.0005,is_distribution = True)
        self.valuenet = network([9,128,128,1],0.0005,)
        self.policynet.apply(init_weights)
        self.memory = memory()

    def storememory(self,state,action,probability,value,reward):
        self.memory.store(state,action,probability,value,reward)
    
    def endtrajectory(self):
        self.memory.storetrajectory()

    def saveparameters (self,policyfile,valuefile):
        self.policynet.saveparameters(policyfile)
        self.valuenet.saveparameters(valuefile)
    
    def loadparameters(self,policyfile,valuefile):
        self.policynet.load(policyfile)
        self.valuenet.load(valuefile)

    def getaction(self,inputstate,show_probs = False):
        state = torch.tensor(np.array([inputstate]))
        distribution = self.policynet(state)
        if(show_probs):
            print(distribution.probs)
        value = self.valuenet(state)
        action = distribution.sample()

        return torch.squeeze(action).item(),\
               torch.squeeze(distribution.log_prob(action)).item(),\
               torch.squeeze(value).item()

    def train(self,iterations,epsilon = 0.2,discount = 0.95):
        data = self.memory.generatetrajectories()
        states_ = [x[0] for x in data]
        actions_ = [x[1] for x in data]
        probs_ = [x[2] for x in data]
        values_  = [x[3] for x in data]
        rewards__ = [x[4] for x in data]
        futurerewards = []
        advantages = []
        

        for i in range(len(rewards__)):

            rewards_ = rewards__[i]
            vals_ = values_[i]
            
            # calculate reward esitmates
            rewards_to_go_ = np.zeros(len(rewards_))  

            for T in range (len(rewards_to_go_)):
                discount_ = 1
                R = 0
                for t in range (T,len(rewards_to_go_)):
                    R += discount_ * rewards_[t]
                    discount_ = discount_ * discount
                rewards_to_go_[T] = R

            futurerewards.append(rewards_to_go_)

            # calculate advantage estimates
            advantages_ = np.zeros(len(rewards_))

            for T in range (len(advantages_) - 1):
                discount_ = 1
                A = 0
                for t in range(T,len(advantages_ ) - 1):
                    A+= discount_ * (rewards_[t] + discount * vals_[t+1] - vals_[t])
                    discount_ = discount_ * discount
                advantages_[T] = A
                
            advantages.append(advantages_)
            

        
        states = torch.tensor(flatten2d(states_))
        oldprobs = torch.tensor(flatten2d(probs_))
        actions = torch.tensor(flatten2d(actions_))
        rewards_to_go = torch.tensor(flatten2d(futurerewards))
        advantages = torch.tensor(flatten2d(advantages))


        # training on policy
        
        for _ in range (iterations):
            self.policynet.optimizer.zero_grad()
            distribution = self.policynet(states) 
            newprobs = distribution.log_prob(actions)
            ratio = (newprobs - oldprobs).exp()
            PPOLoss = -1 * torch.min(advantages * ratio, g(epsilon,advantages)).mean()
            PPOLoss.backward()
            self.policynet.optimizer.step()

        # trainig on value estimator
        for _ in range (iterations):
            self.valuenet.optimizer.zero_grad()
            networkvalues = self.valuenet(states)
            valueloss = ((networkvalues - rewards_to_go) **2).mean()
            weightedvalueloss = valueloss * 0.5
            weightedvalueloss.backward()
            self.valuenet.optimizer.step()
        
        self.memory.clear(clear_trajectories = True)




def maxindex (tensor):
    max = tensor[0]
    maxindex = 0
    for x in range(len(tensor)):
        if tensor[x] > max:
            max = tensor[x]
            maxindex = x
    return maxindex


def restart():
    global state
    global points
    points = []
    state1 = np.array([xdim/3.0,900.0,0.0,0.0, np.float64(randrange(np.pi/4,np.pi/2)),0.0,0.0,np.pi/2,20.0])

def unitvector(angle):
    return np.array([np.cos(angle),np.sin(angle)])
                    
def rotatevector (ang,v):
    return v[0] * unitvector(ang) + v[1] * unitvector(ang + np.pi/2)
        
pygame.init()
textfont = pygame.font.SysFont("UbuntuSans",10)

def drawtext(text, font, color, surface, x, y):
    textobj = font.render(text, True, color)
    textrect = textobj.get_rect()
    textrect.center = (x, y)
    surface.blit(textobj, textrect)

def drawvector (v,p,color):
    pygame.draw.line(screen, color, tuple(newcoordinates(p)), tuple(newcoordinates(p+v)), 1)
    


def getflamevertex (pos,ang,engineangle ):
    v = np.array([np.cos(ang), np.sin(ang)])
    q = rotatevector(-1 * (np.pi/2 - ang), 15 * unitvector(engineangle))
    w = rotate90(q)
    return [
        newcoordinates(pos - ( v * 50)),
        newcoordinates(pos - (v * 50) - q/1.5  + w/2),
        newcoordinates(pos - (v * 50) - q/1.5 -  w/2)
    ]
def getfirevertex(pos,ang,engineangle):
    v = np.array([np.cos(ang), np.sin(ang)])
    q = rotatevector(-1 * (np.pi/2 - ang), 15 * unitvector(engineangle))
    w = rotate90(q)
    return [
        newcoordinates(pos - (v * 50) - q*3),
        newcoordinates(pos - (v * 50) - q/1.5  + w/2),
        newcoordinates(pos - (v * 50) - q/1.5  - w/2)
    ]

    
def help(v):
    vertices = []
    for vertex in v:
        vertices.append(tuple(vertex))
    return vertices

def printpath (path,color = (55,60,50)):
    if len(path) < 2:
        return
    for x in range (len(path) - 1):
        drawvector( path[x+1] - path[x], path[x],color)
    
        
   

screen = pygame.display.set_mode((xdim,ydim))
lasttime = time.time()
backdrop = pygame.Rect(0,0,xdim + 400,ydim)
landingpad = pygame.Rect(xdim/2 - 100,ydim - groundheight,200,groundheight/2)
ground = pygame.Rect(0,ydim - groundheight,xdim,groundheight)
starttime = time.time()
avgdelta = 0
maxdelta = 0

def combine (arr1,arr2):
    out = []
    for x in arr1:
        out.append(x)
    for x in arr2:
        out.append(x)
    return out
        
def extrapolate (state,n):
    if n == 1:
        return [np.array([transition(state,0)[0],transition(state,0)[1]])]
    else:
        return combine([np.array([transition(state,0)[0],transition(state,0)[1]])],extrapolate(transition(state,0),n-1))

def max_action_n_steps(state,n):
  q = 1.4
  if n == 1:
    r = reward(transition(state,1)) + 0.9 * reward(transition(transition(state,1),0)) +   0.9 * reward(transition(transition(transition(state,1),0),0))
    a = 1
    for x in range (3):
      action = x+2
      rtemp = reward(transition(state,action)) + 0.9 * reward(transition(transition(state,action),0)) + 0.9 * reward(transition(transition(transition(state,action),0),0))
      if rtemp > r:
        r = rtemp
        a = action
    return a
  else:
    r = (1/(n ** q)) * reward(transition(state,1)) + max_action_n_steps(transition(state,1),n-1)
    a = 1
    for x in range (3):
      action = x+2
      rtemp = (1/(n ** q)) * reward(transition(state,action)) + max_action_n_steps(transition(state,action),n-1)
      if rtemp > r:
        r = rtemp
        a = action
    return a

def printstate(state):
   position = np.array([state[0],state[1]])
   velocity = np.array([state[2],state[3]])
   angle = state[4]
   engineangle = state[7]
   engine = state[6]
   pygame.draw.rect(screen,(57,54,138),backdrop)
   pygame.draw.rect(screen,(204,102,0),ground)
   pygame.draw.polygon(screen, (0,0,0),help(getvertex(position, angle)))
   pygame.draw.polygon(screen,(100,100,100),help(getflamevertex(position,angle,engineangle)))
   pygame.draw.polygon(screen, (131,131,131),help(getvertex(np.array([xdim + 150,500]), angle)))
   pygame.draw.polygon(screen,(77,77,77),help(getflamevertex(np.array([xdim + 150,500]),angle,engineangle)))
   pygame.draw.rect(screen,(90,77,77),landingpad)
   if engine == 1:
       pygame.draw.polygon(screen,(255,100,0),help(getfirevertex(position,angle,engineangle)))
       pygame.draw.polygon(screen,(255,100,0),help(getfirevertex(np.array([xdim + 150,500]),angle,engineangle)))



clock = pygame.time.Clock()



def testrun (pilot):
    points = [np.array([xdim/2.5,700])]
    state =  np.array([xdim/2.5,700.0,0.0,0.0, np.float64(randrange(np.pi/4,np.pi/3)),0.0,0.0,np.pi/2,20.0]) 
    while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
            position = np.array([state[0],state[1]])
            velocity = np.array([state[2],state[3]])
            angle = state[4]
            engineangle = state[7]
            engine = state[6]
            points.append(np.array([state[0],state[1]]))
            a = pilot.getaction(state)[0] + 1
            printstate(state)
            state = transition(state,a)
            for x in getvertex(position,angle):
                if x[1] >= 900 or x[1] <= 0 or x[0] >= xdim or x[0] <= 0:
                    print("testrun was terminated")
                    return
            drawvector(velocity,np.array([xdim + 150,500]),(255,0,0))
            drawvector(gravity,np.array([xdim + 150,500]),(255,0,0))
            printpath(points,(255,255,255))
            printpath(extrapolate(state,300),(254,2,25))
            pygame.display.update()
            clock.tick(60)
            


restart()

pilot = agent()

n = 0    
while True:
    print("trial number : " + str(n))
    state =  np.array([xdim/2.5,700.0,0.0,0.0, np.float64(randrange(np.pi/4,np.pi/3)),0.0,0.0,np.pi/2,20.0]) 
    if n % 10 == 0 and n != 0 :
        pilot.getaction(state,show_probs = True)
        testrun(pilot)
    n += 1
    done = False
    for _ in range (7):
        state =  np.array([xdim/2.5,700.0,0.0,0.0, np.float64(randrange(np.pi/3,np.pi/2)),0.0,0.0,np.pi/2,20.0]) 
        while not done:
            action,prob,val = pilot.getaction(state)
            nextstate = transition(state,action+1)
            reward_ = reward(nextstate)
            for x in getvertex(np.array([state[0],state[1]]),state[4]):
                if x[1] >= 900 or x[1] <= 0 or x[0] >= xdim or x[0] <= 0:
                    done = True
                    break
            pilot.storememory(state,action,prob,val,reward_,)
            state = nextstate
        pilot.endtrajectory()
    pilot.train(10)



    # while True :
    # # for state in visual:
    #     position = np.array([state[0],state[1]])
    #     velocity = np.array([state[2],state[3]])
    #     angle = state[4]
    #     engineangle = state[7]
    #     engine = state[6]
    #     points.append(np.array([state[0],state[1]]))
    #     a = 0
    #     printstate(state)
    #     for event in pygame.event.get():
    #         if event.type == QUIT:
    #             pygame.quit()
    #             sys.exit()
    #         if event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_ESCAPE:
    #                 points = [np.array([xdim/2.5,900])]
    #                 state =  np.array([xdim/2.5,900.0,0.0,0.0, np.float64(randrange(np.pi/3,np.pi/2)),0.0,0.0,np.pi/2,20.0])       
    #             if event.key == pygame.K_f:
    #                 a = 4
    #             if event.key == pygame.K_LEFT:
    #                 a = 2
    #             if event.key == pygame.K_RIGHT:
    #                 a = 3
    #             if event.key == pygame.K_SPACE:
    #                 if pause:
    #                     pause = False
    #                 else:
    #                     pause = True
    #     if not pause:
    #         state = transition(state,a)
    #     for x in getvertex(position,angle):
    #         if x[1] >= 900:
    #             points = [np.array([xdim/2.5,900])]
    #             state =  np.array([xdim/2.5,900.0,0.0,0.0, np.float64(randrange(np.pi/3,np.pi/2)),0.0,0.0,np.pi/2,20.0])
    #     drawvector(velocity,np.array([xdim + 150,500]),(255,0,0))
    #     drawvector(gravity,np.array([xdim + 150,500]),(255,0,0))
    #     printpath(points,(150,150,150))
    #     printpath(extrapolate(state,300),(254,2,25))
    #     pygame.display.update()
    #     clock.tick(60)
            


        

        

    
    
       
    


# Save the model parameters
#torch.save(policynet.state_dict(), file_path)

#policynet.load_state_dict(torch.load(file_path))







clock = pygame.time.Clock()





restart()



    
