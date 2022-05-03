"""Solves an instance.

Modify this file to implement your own solvers.

For usage, run `python3 solve.py --help`.
"""

import argparse
from ast import Or
from calendar import c
import time
# from imp import source_from_cache
from pathlib import Path
from turtle import pos
from random import sample
# from selectors import EpollSelector
from typing import Callable, Dict
from xml.etree.ElementPath import find
import math
import time
#test
from instance import Instance
from point import Point
from solution import Solution
from file_wrappers import StdinFileWrapper, StdoutFileWrapper
from itertools import chain, combinations


def solve_naive(instance: Instance) -> Solution:

    return Solution(
        instance=instance,
        towers=instance.cities,
    )






def createpossilbesols(city, size):
    x = city.x
    y = city.y
    pos = []
    xrang = (range(x - 2, x + 3))
    yrang = (range(y - 2, y + 3))

    for i in xrang:
        for j in yrang:
            if (0 <= i < size and 0 <= j < size ):
                pos.append(Point(i,j))

    return pos


def poscitiesincircle(tower, allcities):
    posciti =[]
    count = 0
    for city in allcities:
       
        if (city.x <= tower.x + 2 and city.x >= tower.x -2 and city.y <= tower.y + 2 and city.y >= tower.y -2) or (city.x == tower.x+3 and city.y ==tower.y) or (city.x == tower.x-3 and city.y ==tower.y) or (city.x == tower.x and city.y ==tower.y +3 )or( city.x == tower.x and city.y ==tower.y -3):
            count +=1
            posciti += [city]
    return (posciti,count)

#finds the farthest city from a city

def closest_city(city,allcities):
    x = city.x
    y = city.y
    min_dist = math.inf
    ret = None
    for close in allcities:
        dist_sq = ((close.x -x)**2) +((close.y -y)**2)
        if min_dist >    dist_sq:
            min_dist = dist_sq
            ret = close
    return ret

def find_path(city,all_cities):
    order = [city]
    length = len(all_cities)
    while len(order) < length:
        all_cities.remove(city)
        city = closest_city(city,all_cities)
        order +=[city]
    return order


def solve_algo_greedy(Instance: Instance) -> Solution:

    start = time.time()
    allcities = Instance.cities.copy()
    size = Instance.grid_side_length
    towers2 = [] 
    leftrowmin = size

    max_tower = None
    max_tower_val = 0
    allsol = []
    next_del = None
    solution = False
    tower_cover = {}
    for city in allcities:
        allsol += createpossilbesols(city,size)
    #print('allsol',allsol[0])
    while solution == False:
       # print(towers2)
        max_tower = None
        max_tower_val = 0
        next_del =  None
        for t in allsol:
            #print(t)
            val = poscitiesincircle(t,allcities)
            #print('allcities', allcities)
            #print(val)
            #print(allsol)
            
            if len(allcities) <3:
                for x in allcities:
                    j = createpossilbesols(x,size)
                    
                   # if  t in j:
                        
                        #print(x,t,val)
                        #print('available')
               

                
            if val[1] > max_tower_val:
                
                max_tower_val = val[1]
                max_tower = t
                next_del= val[0]
        #print('done', allcities)
        tower_cover[max_tower] = next_del
        for x in next_del:
            if x in allcities:
                allcities.remove(x)
        #print(towers2)
        if max_tower not in towers2:

            towers2+=[max_tower]
        possiblesol = Solution(instance = Instance ,towers = towers2)
        
        solution = possiblesol.valid()
    # valid  ={}
    # for xor in towers2:
    #     covered = tower_cover[xor]
    #     values= createpossilbesols(xor,size)
    #      valid_towers= []
    #     for xand in values:
    #     #     solu = poscitiesincircle(xand,values)
    #     #     if len(solu[0]) == len(covered):
    #     #         valid_towers +=[xand]
    #     # print(valid_towers)
    #     valid[xor] = values
    # count  = 0
    # oldtowers = towers2.copy()
    # baseline = possiblesol.penalty()
    # while (count < 10):
    #     newlist = []
    #     count += 1
    #     for i in valid.keys():
    #         newpos = sample(valid[i], 1) 
    #         print (newpos)
    #     newsol = Solution(instance = Instance ,towers = newlist)
    #     if (newsol.valid and newsol.penalty() < baseline):
    #         possiblesol = newsol
    #         baseline = newsol.penalty()
    #         oldtowers = newlist

    start = time.time()
    size = Instance.grid_side_length
    baseline = possiblesol.penalty()
    newtowers = towers2.copy()
    oldtowers = newtowers.copy()
    count  = 0


    for t in towers2:
        #need to keep track of previous towers
        oldtowers = newtowers.copy()
        #hypothetical removal of the new tower t
        indx = newtowers.index(t)
        newtowers.remove(t)

        possysoo = Solution(instance=Instance, towers=newtowers)
        #Check if valid and penalty is less than baseline
        penaltycities = possysoo.validtweaked()
        if (len(newtowers) == 1):
            break
        if len(penaltycities) == 0 and possysoo.penalty() < baseline and possysoo.valid():
            baseline = possysoo.penalty()
            print('deleted tower city')
            continue
        else:
            #Move around Towers
            mintowersol = None
            towerremoved = None
            #newtowers = oldtowers
            newnewtowers = oldtowers.copy()
            for cit in penaltycities:
                #towertoreplace = closest_city(cit, newnewtowers)
                #newnewtowers.remove(towertoreplace)
                allpositions = createpossilbesols(cit, size)
                for pos in allpositions:
                   # newtowers = newnewtowers
                    #newtowers.append(pos)
                    newnewtowers[indx] = pos
                    possysoo = Solution(instance=Instance, towers=newnewtowers)
                    pen = possysoo.penalty()
                    if possysoo.valid() and pen < baseline:
                        baseline = pen
                        mintowersol = pos
                        #towerremoved = towertoreplace
            if mintowersol == None:
                newtowers = oldtowers
                #print('no sol')
                continue
            else:
                newtowers = oldtowers
                newtowers[indx] = mintowersol
               # print('better sol')
                #print(towerremoved)
                #newtowers.remove(towerremoved)
                #newtowers.append(mintowersol)
                
    end = time.time()

    print('time taken:', math.ceil(end-start), "seconds",'# towers: ',len(newtowers))

    return Solution(instance=Instance, towers=newtowers)


    

            
        


    return possiblesol
       # print(solution)

    #possiblesol = solution
    baseline = possiblesol.penalty()
    solutionfound = True
    newtowers = towers2.copy()
    oldtowers = newtowers.copy()
    count  = 0    
    for t in towers2:
        count += 1
        #print(count)
        #need to keep track of previous towers
        oldtowers = newtowers.copy()
        #hypothetical removal of the new tower t
        indx = newtowers.index(t)
        newtowers.remove(t)

        possiblesol = Solution(instance=Instance, towers=newtowers)
        #Check if valid and penalty is less than baseline
        penaltycities = possiblesol.validtweaked()
        if len(penaltycities) == 0 and possiblesol.penalty() < baseline:
            baseline = possiblesol.penalty()
          #  print('deleted tower city')
            continue
        else:
            #Move around Towers
            mintowersol = None
            towerremoved = None
            #newtowers = oldtowers
            newnewtowers = oldtowers.copy()
            for cit in penaltycities:
                #towertoreplace = closest_city(cit, newnewtowers)
                #newnewtowers.remove(towertoreplace)
                allpositions = createpossilbesols(cit, size)
                for pos in allpositions:
                   # newtowers = newnewtowers
                    #newtowers.append(pos)
                    newnewtowers[indx] = pos
                    possiblesol = Solution(instance=Instance, towers=newnewtowers)
                    pen = possiblesol.penalty()
                    if possiblesol.valid() and pen < baseline:
                        baseline = pen
                        mintowersol = pos
                        #towerremoved = towertoreplace
            if mintowersol == None:
                newtowers = oldtowers
                #print('no sol')
                continue
            else:
                newtowers = oldtowers
                newtowers[indx] = mintowersol
               # print('better sol')
                #print(towerremoved)
                #newtowers.remove(towerremoved)
                #newtowers.append(mintowersol)
                
    end = time.time()

    print('time taken:', math.ceil(end-start), "seconds",'# towers: ',len(newtowers))

    return Solution(instance=Instance, towers=newtowers)


def solve_algorithm(Instance: Instance) -> Solution:
    start = time.time()
    allcities = Instance.cities
    size = Instance.grid_side_length
    towers = [] 
    leftrowmin = size
    # Keys with cities as num and with values as set of possible towers
    city_key = {}
    rightrowmax = 0
    
    # 
    memo_sol = {}
    cityrows = {}
    covered=[]
    for i in range(size):
        cityrows[i] =[]
   
    counter = 0
    for city in allcities:
        cityrows[city.x] += [city]
        city_key[set(city)] = counter
        counter +=1
        leftrowmin  = min(leftrowmin, city.x)
        rightrowmax = max(rightrowmax, city.x)
    
    
    possiblesol= False
    cur_city = cityrows[leftrowmin][0]
    incr = 0
    #we find the path of cities to visit from find_path function.
    path = find_path(cur_city,allcities)
    while  possiblesol == False:   
        min_tower_val= 0
        #I created a variable called no_tower to help with the base case when we have no tower placed down. When choosing the first tower our first algorithm chooses the location that maximizes coverage of cities
        no_tower = False
        for postower in createpossilbesols(cur_city, size):
            min_tower = postower
            val_tower = poscitiesincircle(postower, allcities)
            
            if len(towers) ==0:
                no_tower = True
            else:
                cost_tower = poscitiesincircle(postower,towers)[1]
            
            cost = 0

            if no_tower==False:
                cost = (val_tower[1] * -1 ) + (cost_tower)
            else:
                cost = (val_tower[1] * -1 )
            if min_tower_val >= cost:
                min_tower_val = cost
                min_tower = postower
                city_cov = val_tower[0]
            key = []
            for x in city_cov:
                key += [city_key(x)]
            if set(key) in memo_sol.keys():
                memo_sol[set(key)] = min(cost,memo_sol[key])
            else:
                memo_sol[set(key)] =cost
         
        covered += city_cov
        incr +=1
        for x in covered:
            path.remove(x)
        
        cur_city= path[incr]
        

        if len(towers) ==0:
            towers += [min_tower]
            no_tower = False
        else:
            towers += [min_tower]

        possiblesol = Solution(instance=Instance, towers=towers).valid()

 
    return Solution(instance=Instance, towers=towers)


    def validtweaked(self):
        """Determines whether a solution is valid.

        A solution is valid for a problem instance if its towers cover all
        cities in the instance, all towers are in bounds, and there are no
        duplicate towers.
        """

        badcities = []
        for city in self.instance.cities:
            for tower in self.towers:
                if Point.distance_obj(city, tower) <= self.instance.coverage_radius:
                    break
            else:
                badcities.append(city)

        return badcities

            
    
            


def solve_other_algorithm(Instance: Instance) -> Solution:
    start = time.time()
    size = Instance.grid_side_length
    towers = Instance.cities.copy()
    possiblesol = Solution(instance=Instance, towers=towers)
    baseline = possiblesol.penalty()
    solutionfound = True
    newtowers = towers.copy()
    oldtowers = newtowers.copy()
    count  = 0
    #The new possible solution needs to be valid and have a more optimal penalty value
    for t in towers:
        count += 1
        #print(count)
        #need to keep track of previous towers
        oldtowers = newtowers.copy()
        #hypothetical removal of the new tower t
        indx = newtowers.index(t)
        newtowers.remove(t)

        possiblesol = Solution(instance=Instance, towers=newtowers)
        #Check if valid and penalty is less than baseline
        penaltycities = possiblesol.validtweaked()
        if len(penaltycities) == 0 and possiblesol.penalty() < baseline:
            baseline = possiblesol.penalty()
          #  print('deleted tower city')
            continue
        else:
            #Move around Towers
            mintowersol = None
            towerremoved = None
            #newtowers = oldtowers
            newnewtowers = oldtowers.copy()
            for cit in penaltycities:
                #towertoreplace = closest_city(cit, newnewtowers)
                #newnewtowers.remove(towertoreplace)
                allpositions = createpossilbesols(cit, size)
                for pos in allpositions:
                   # newtowers = newnewtowers
                    #newtowers.append(pos)
                    newnewtowers[indx] = pos
                    possiblesol = Solution(instance=Instance, towers=newnewtowers)
                    pen = possiblesol.penalty()
                    if possiblesol.valid() and pen < baseline:
                        baseline = pen
                        mintowersol = pos
                        #towerremoved = towertoreplace
            if mintowersol == None:
                newtowers = oldtowers
                #print('no sol')
                continue
            else:
                newtowers = oldtowers
                newtowers[indx] = mintowersol
               # print('better sol')
                #print(towerremoved)
                #newtowers.remove(towerremoved)
                #newtowers.append(mintowersol)
                
    end = time.time()

    print('time taken:', math.ceil(end-start), "seconds",'# towers: ',len(newtowers))

    return Solution(instance=Instance, towers=newtowers)



    

                
                

        






SOLVERS: Dict[str, Callable[[Instance], Solution]] = {
    "naive": solve_naive,
    'algo': solve_algo_greedy
}


# You shouldn't need to modify anything below this line.
def infile(args):
    if args.input == "-":
        return StdinFileWrapper()

    return Path(args.input).open("r")


def outfile(args):
    if args.output == "-":
        return StdoutFileWrapper()

    return Path(args.output).open("w")


def main(args):
    with infile(args) as f:
        instance = Instance.parse(f.readlines())
        solver = SOLVERS[args.solver]
        solution = solver(instance)
        assert solution.valid()
        with outfile(args) as g:
            print("# Penalty: ", solution.penalty(), file=g)
            solution.serialize(g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve a problem instance.")
    parser.add_argument("input", type=str, help="The input instance file to "
                        "read an instance from. Use - for stdin.")
    parser.add_argument("--solver", required=True, type=str,
                        help="The solver type.", choices=SOLVERS.keys())
    parser.add_argument("output", type=str,
                        help="The output file. Use - for stdout.",
                        default="-")
    main(parser.parse_args())