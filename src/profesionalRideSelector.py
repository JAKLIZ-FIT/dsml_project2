# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 18:00:53 2023

@author: jzilk
"""

import random

rides = ["Wadon","Green Fire","Gold Star","ISS","Fjord Fun",
         "Austrian white water","Netpun","Pirates of Bavaria","Dino Garden",
         "Ferry Boat"]
         
print(len(rides))

rides.pop(1)

for i in range(6):
    aux = random.choice(rides)
    rides.remove(aux)
    print(aux)
    
"""
Jiri:
    ISS
    Austrian white water

Shanaya:
    Ferry Boat
    Netpun

Zalan:
    Fjord Fun
    Dino Garden
"""