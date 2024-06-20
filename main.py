import pygame, sys, time
import numpy as np
from pygame.locals import QUIT
from random import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class net(nn.Module):
 
    def __init__(self, n_observations, n_actions):
        super(net, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128, dtype=torch.float64)
        self.layer2 = nn.Linear(128, 128, dtype=torch.float64)
        self.layer3 = nn.Linear(128, n_actions, dtype=torch.float64)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

policynet = net(9, 4).to(device)
targetnet = net(9, 4).to(device)
optimizer = optim.Adam(policynet.parameters(), lr=0.001)
lf = nn.MSELoss()

def epsilon (steps_done):
    if steps_done < 200:
        return 0.5
    if steps_done < 400:
        return 0.4
    if steps_done < 600:
        return 0.2
    if steps_done < 800:
        return 0.1
    return 0

trials = 0


def randgreedy(state, steps_done):
    n = random()
    e = epsilon(steps_done)
    if n < (1 - e):
        with torch.no_grad():
            v = policynet(torch.tensor(state))
            return maxindex(policynet(torch.tensor(state)))
    return np.random.choice(np.array([1,2,3,4]))

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
        if state [0] <= 0 or state[0] >= xdim or state [1] <= 100 or state[1] >= ydim:
            return out
        action = randgreedy(state, trials)
        nextstate = transition(state,action)
        value = reward(nextstate)
        out.append((state,action,value,nextstate))
        state = nextstate
    return out
        

    
    























clock = pygame.time.Clock()

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
    

gravity = np.array([0,-18])

def rotate90 (v):
    return np.array([-1 * v[1],v[0]])

def newcoordinates(p):
    return np.array([p[0],ydim-p[1]])

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
def getengineforce(ang,engineang):
    v = np.array([np.cos(ang), np.sin(ang)])
    return [
        np.array(v * np.sin(engineangle) * engineforce),
        (engineforce/(2 * np.pi * 50) * np.cos(engineang))
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
    
        

"""
state vector defined :
0 positoinx 
1 postitiony
2 velocityx
3 velocityy
4 angle
5 anglevelocity
6 engine
7 engineangle
8 fuel

action defined :

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
    angle = state[4]
    for x in getvertex(np.array([positionx,positiony]),angle):
     if x[1] >= 900 or x[0] <= 0 or x[0] >= xdim:
         return 0
    velocityx = state[2]
    velocityy = state[3]
    anglevelocity = state[5]
    fuel = state[8]
    out = 0
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

        
        
        
        
        
        
        
        
        
        

screen = pygame.display.set_mode((xdim,ydim))
lasttime = time.time()
backdrop = pygame.Rect(0,0,xdim + 400,ydim)
landingpad = pygame.Rect(xdim/2 - 100,ydim - groundheight,200,groundheight/2)
ground = pygame.Rect(0,ydim - groundheight,xdim,groundheight)
starttime = time.time()
iterations = 0
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


restart()


visual = []
temp = trajectory(np.array([xdim/3.0,900.0,0.0,0.0, randrange(np.pi/4,np.pi/2),0.0,0.0,np.pi/2,20.0]),1000)
print(temp)
for x in temp:
    visual.append(x[0])
     
 

for state in visual:
   position = np.array([state[0],state[1]])
   velocity = np.array([state[2],state[3]])
   angle = state[4]
   engineangle = state[7]
   engine = state[6]
   points.append(np.array([state[0],state[1]]))
   iterations += 1
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

n = 0
trajectories = []
while n < 10:

    trajectories.appendtrajectory(np.array([xdim/3.0,900.0,0.0,0.0, randrange(np.pi/4,np.pi/2),0.0,0.0,np.pi/2,20.0]),1000)
        
    
    
    