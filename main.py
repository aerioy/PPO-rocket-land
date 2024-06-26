import pygame, sys, time
import numpy as np
from pygame.locals import QUIT
import random as r
from random import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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
    return [
        newcoordinates(pos + v * 70),
        newcoordinates(pos + (50 * v + 10 * w)),
        newcoordinates(pos + (-50 * v + 10 * w)),
        newcoordinates(pos + (-50 * v + (-10) * w)),
        newcoordinates(pos + (50 * v + (-10)  *w))
    ]
    
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

def magnitude(array):
    mag = np.sqrt(np.sum(array ** 2)) 
    return mag



def reward (state):
    positionx = state[0]
    positiony = state[1]
    position = np.array([positionx,positiony])
    angle = state[4]
    for x in getvertex(np.array([positionx,positiony]),angle):
     if x[1] >= 900 or x[0] <= 0 or x[0] >= xdim:
         return 0
    velocityx = state[2]
    velocityy = state[3]
    anglevelocity = state[5]
    fuel = state[8]
    out = (-1 * magnitude(position - np.array([xdim/2,100]))) ** 2 + fuel + (-1 * abs(velocityy)) ** 9  + 1/3 * (-1 * abs(velocityx)) + (-1 * abs(angle - np.pi/2)) ** 5
    return out


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

        
        
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class net(nn.Module):
 
    def __init__(self, n_observations, n_actions):
        super(net, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128, dtype=torch.float64)
        self.layer2 = nn.Linear(128, 128, dtype=torch.float64)
        self.layer3 = nn.Linear(128, n_actions, dtype=torch.float64)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

policynet = net(9, 4).to(device)
targetnet = net(9, 4).to(device)
targetnet.load_state_dict(policynet.state_dict())

optimizer = optim.Adam(policynet.parameters(), lr=0.00001)
lf = nn.MSELoss()

def epsilon (steps_done):
    if steps_done < 200:
        return 0.4
    if steps_done < 400:
        return 0.2
    if steps_done < 600:
        return 0.1
    if steps_done < 800:
        return 0.01
    return 0

trials = 0
discount = 0.99

def randgreedy(state, steps_done):
    n = random()
    e = epsilon(steps_done)
    if n < (1 - e):
        with torch.no_grad():
            return maxindex(policynet(torch.tensor(state)))
    return np.random.choice(np.array([1.0,2.0,3.0,4.0]))

def maxindex (tensor):
    max = tensor[0]
    maxindex = 0
    for x in range(len(tensor)):
        if tensor[x] > max:
            max = tensor[x]
            maxindex = x
    return maxindex


def trajectory(initialstate,maxlength):
    global trials
    trials += 1 
    state = initialstate
    out = []
    for x in range(maxlength):
        for w in (getvertex(np.array([state[0],state[1]]),state[4])):   
           if w[0] <= 0 or w[0] >= xdim or w[1] >= 900 or w[1] <= 0:
              return out
        action = randgreedy(state, trials)
        nextstate = transition(state,action)
        value = reward(nextstate)
        out.append((state,action,value,nextstate))
        state = nextstate
    return out
      
# (state, action, reward, nextstate)
def gettrainingbatch(trajectory):
    if len(trajectory) < 128:
        minibatch = trajectory
        statetensor = []
        actiontensor = []
        ytensor = torch.zeros(len(trajectory))
        
    else:   
        indices = np.random.choice(len(trajectory), size=128, replace=False)
        minibatch = list(map(lambda x : trajectory[x],indices))
        statetensor = []
        actiontensor = []
        ytensor = torch.zeros(128)
    for x in range(len(minibatch)):
        with torch.no_grad():
            v = targetnet.forward(torch.tensor(minibatch[x][3])).max()
        ytensor[x] = minibatch[x][2] + discount * v
        statetensor.append(minibatch[x][0])
        actiontensor.append(minibatch[x][1])
    return (torch.tensor(np.array(statetensor)),torch.tensor(np.array(actiontensor)),ytensor)    
        
        
        
        
        
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
   pygame.draw.rect(screen,(0,100,255),backdrop)
   pygame.draw.rect(screen,(0,255,0),ground)
   pygame.draw.polygon(screen, (0,0,0),help(getvertex(position, angle)))
   pygame.draw.polygon(screen,(150,150,150),help(getflamevertex(position,angle,engineangle)))
   pygame.draw.polygon(screen, (255,255,255),help(getvertex(np.array([xdim + 150,500]), angle)))
   pygame.draw.polygon(screen,(150,150,150),help(getflamevertex(np.array([xdim + 150,500]),angle,engineangle)))
   pygame.draw.rect(screen,(77,77,77),landingpad)
   if engine == 1:
       pygame.draw.polygon(screen,(255,100,0),help(getfirevertex(position,angle,engineangle)))
       pygame.draw.polygon(screen,(255,100,0),help(getfirevertex(np.array([xdim + 150,500]),angle,engineangle)))



clock = pygame.time.Clock()





restart()


            
while trials <= 1000:
    print("TRIAL NUMBER : " + str(trials))
    trialpath= trajectory(np.array([xdim/3.0,900.0,0.0,0.0, randrange(np.pi/4,np.pi/2),0.0,0.0,np.pi/2,20.0]),1000)
    batch = gettrainingbatch(trialpath)
    states = batch[0]
    actions = batch[1]
    y = batch[2]
    loss = 0
    optimizer.zero_grad()
    for j in range (len(batch[0])):
        loss +=  lf(y[j],policynet(states[j])[int(actions[j] - 1)]) 
    avgloss = loss / len(batch[0])
    avgloss.backward()
    optimizer.step
    if trials % 50 == 0:
        targetnet.load_state_dict(policynet.state_dict())
    if trials % 200 == 0:
        points = [np.array([500,1000])]
        visual = []
        print(trialpath)
        for x in trialpath:
            visual.append(x[0])
        for state in visual:
            position = np.array([state[0],state[1]])
            velocity = np.array([state[2],state[3]])
            angle = state[4]
            engineangle = state[7]
            engine = state[6]
            points.append(np.array([state[0],state[1]]))
            a = 0
            printstate(state)
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        restart()        
                    if event.key == pygame.K_f:
                        a = 4
                    if event.key == pygame.K_LEFT:
                        a = 2
                    if event.key == pygame.K_RIGHT:
                        a = 3
                    if event.key == pygame.K_SPACE:
                        if pause:
                            pause = False
                        else:
                            pause = True
            #    if not pause:
            #      state = transition(state,a)
            #    for x in getvertex(position,angle):
            #     if x[1] >= 900 or x[0] <= 0 or x[0] >= xdim:
            #         restart()
            drawvector(velocity,np.array([xdim + 150,500]),(255,0,0))
            drawvector(gravity,np.array([xdim + 150,500]),(255,0,0))
            printpath(points)
            printpath(extrapolate(state,300),(255,0,0))
            pygame.display.update()
            clock.tick(60)
            


        

        

    
    
       
    


# Save the model parameters
#torch.save(policynet.state_dict(), file_path)

#policynet.load_state_dict(torch.load(file_path))







clock = pygame.time.Clock()





restart()



    
