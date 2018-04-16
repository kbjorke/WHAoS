#!/usr/bin/env python

# Tested with http://tools.druchii.net/AoS-Combat-Calculator.php

# By: Kristian Bjoerke

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scpspes
from itertools import product

#def prob_sum(dice_number, sides, target):
#    """
#    Probability to get a sum (target) when rolling a number (dice_number)
#    of a certain dice type (sides).
#    """
#    rollAmount = 0
#    targetAmount = 0
#    for i in product(range(1,sides+1), repeat=dice_number):
#        thisSum = 0
#        rollAmount += 1
#        for j in i:
#            thisSum += j
#        if thisSum == target:
#            targetAmount += 1
#    odds = float(targetAmount) / rollAmount
#    return odds

def dice_prob(dice_sides):
    """
    Probability for each side of a (dice_sides) sided dice.
    """
    return 1.0/dice_sides

def prob_success(dice_sides, target,rr1=False,rr2=False,rrf=False):
    """
    Probability to get success when rolling a (dice_sides) sided dice
    and requireing to get (target) value or higher.
    """
    prob = 0
    for i in range(target,dice_sides+1):
        prob += dice_prob(dice_sides)

    for i in range(target,dice_sides+1):
        if rr1 and target > 1:
            prob += dice_prob(dice_sides)**2
        if rr2 and target > 2:
            prob += 2*dice_prob(dice_sides)*dice_prob(dice_sides)
        if rrf and target > 1:
            prob += (target-1)*dice_prob(dice_sides)*dice_prob(dice_sides)
    return prob

def prob_Nsuccess(dice_number, dice_sides, number_success, target,rr1=False,rr2=False,rrf=False):
    """
    Probability to get N (number_success) successes when rolling
    a number of dice (dice_number) that are (dice_sides) sided dice.
    Success is defined as getting (target) value or higher on dice.
    """
    prob = scpspes.binom(dice_number, number_success)
    prob *= prob_success(dice_sides, target,rr1,rr2,rrf)**(number_success)
    prob *= (1 - prob_success(dice_sides, target,rr1,rr2,rrf))**(dice_number - number_success)
    return prob

def prob_fail(dice_sides, target,rr1=False,rr2=False,rrf=False):
    """
    Probability to fail when rolling a (dice_sides) sided dice
    and requireing to get (target) value or higher.
    """
    prob = 0
    for i in range(1, target):
        prob += dice_prob(dice_sides)
    
    for i in range(1, target):
        if rr1 and target > 1:
            prob -= dice_prob(dice_sides)**2
        if rr2 and target > 2:
            prob -= 2*dice_prob(dice_sides)*dice_prob(dice_sides)
        if rrf and target > 1:
            prob -= (target-1)*dice_prob(dice_sides)*dice_prob(dice_sides)
    return prob

def prob_Nfail(dice_number, dice_sides, number_fail, target,rr1=False,rr2=False,rrf=False):
    """
    Probability to get N (number_fail) fals when rolling
    a number of dice (dice_number) that are (dice_sides) sided dice.
    Success is defined as getting (target) value or higher on dice.
    """
    #prob = scpspes.binom(dice_number, number_success)
    #prob *= prob_success(dice_sides, target)**(number_success)
    #prob *= (1 - prob_success(dice_sides, target))**(dice_number - number_success)
    prob = scpspes.binom(dice_number, number_fail)
    prob *= prob_fail(dice_sides, target,rr1,rr2,rrf)**(number_fail)
    prob *= (1 - prob_fail(dice_sides, target,rr1,rr2,rrf))**(dice_number - number_fail)
    return prob

def prob_hits(attacks, to_hit):
    """
    Discrete probabilties for number of hits when making a number 
    of (attacks) with a weapon with a (to_hit) value.
    """
    if reroll_hit_1:
        return [prob_Nsuccess(attacks,6,x,to_hit,rr1=True) for x in range(0,attacks+1)] 
    if reroll_hit_2:
        return [prob_Nsuccess(attacks,6,x,to_hit,rr2=True) for x in range(0,attacks+1)] 
    if reroll_hit_f:
        return [prob_Nsuccess(attacks,6,x,to_hit,rrf=True) for x in range(0,attacks+1)] 
    else:
        return [prob_Nsuccess(attacks,6,x,to_hit) for x in range(0,attacks+1)] 


def expec_hits(attacks, to_hit):
    """
    Expectation value for number of hits when making a number
    of (attacks) with a weapon with a (to_hit) value.
    """
    if reroll_hit_1:
        return attacks*prob_success(6,to_hit,rr1=True)
    elif reroll_hit_2:
        return attacks*prob_success(6,to_hit,rr2=True)
    elif reroll_hit_f:
        return attacks*prob_success(6,to_hit,rrf=True)
    else:
        return attacks*prob_success(6,to_hit)

def prob_wounds(attacks, to_hit, to_wound):
    """
    Discrete probabilties for number of wounds when making a number 
    of (attacks) with a weapon with a (to_hit) value and (to_wound) value.
    """
    phits = prob_hits(attacks, to_hit)
    N = len(phits)
    prob = [0]*N
    for i in range(0,N):
        for x in range(0,N):
            if reroll_wound_1:
                prob[i] += phits[x]*prob_Nsuccess(x,6,i,to_wound,rr1=True)
            elif reroll_wound_2:
                prob[i] += phits[x]*prob_Nsuccess(x,6,i,to_wound,rr2=True)
            elif reroll_wound_f:
                prob[i] += phits[x]*prob_Nsuccess(x,6,i,to_wound,rrf=True)
            else:
                prob[i] += phits[x]*prob_Nsuccess(x,6,i,to_wound)
    return prob

def expec_wounds(attacks, to_hit, to_wound):
    """
    Expectation value for number of hits when making a number
    of (attacks) with a weapon with a (to_hit) value.
    """
    if reroll_wound_1:
        return expec_hits(attacks, to_hit)*prob_success(6,to_wound,rr1=True)
    elif reroll_wound_2:
        return expec_hits(attacks, to_hit)*prob_success(6,to_wound,rr2=True)
    elif reroll_wound_f:
        return expec_hits(attacks, to_hit)*prob_success(6,to_wound,rrf=True)
    else:
        return expec_hits(attacks, to_hit)*prob_success(6,to_wound)

def damage_list(attacks, damage):
    """
    """
    return [x*damage for x in range(0,attacks+1)]

def expec_damage(attacks, to_hit, to_wound, damage):
    """
    Expectation damage when making a number of (attacks) 
    with a weapon with a (to_hit) value.
    """
    #return damage*expec_hits(attacks, to_hit)*prob_success(6,to_wound)
    return damage*expec_wounds(attacks, to_hit, to_wound)

def prob_damage_saves(attacks, to_hit, to_wound, damage, save):
    """
    Discrete probabilties for damage when making a number 
    of (attacks) with a weapon with a (to_hit) hit value and 
    a (to_wound) wound value agains enemy with a (save) save value.
    """
    phits = prob_hits(attacks, to_hit)
    prob_dmg = [0]*len(phits)
    prob_wnd = prob_wounds(attacks, to_hit, to_wound)
    for i in range(0,len(prob_wnd)):
        for x in range(0,len(prob_wnd)):
            if reroll_saves_1:
                prob_dmg[i] += prob_wnd[x]*prob_Nfail(x,6,i,save,rr1=True)
            elif reroll_saves_2:
                prob_dmg[i] += prob_wnd[x]*prob_Nfail(x,6,i,save,rr2=True)
            elif reroll_saves_f:
                prob_dmg[i] += prob_wnd[x]*prob_Nfail(x,6,i,save,rrf=True)
            else:
                prob_dmg[i] += prob_wnd[x]*prob_Nfail(x,6,i,save)
    return prob_dmg

def expec_damage_saves(attacks, to_hit, to_wound, damage, save):
    """
    Expectation damage when making a number of (attacks) with 
    a weapon with a (to_hit) hit value and a (to_wound) wound 
    value agains enemy with a (save) save value.
    """
    #return damage*expec_hits(attacks, to_hit)*prob_success(6,to_wound)*prob_fail(6,save)
    if reroll_wound_1:
        return damage*expec_wounds(attacks, to_hit,to_wound)*prob_fail(6,save,rr1=True)
    elif reroll_wound_2:
        return damage*expec_wounds(attacks, to_hit,to_wound)*prob_fail(6,save,rr2=True)
    elif reroll_wound_f:
        return damage*expec_wounds(attacks, to_hit,to_wound)*prob_fail(6,save,rrf=True)
    else:
        return damage*expec_wounds(attacks, to_hit,to_wound)*prob_fail(6,save)

def prob_damage_saves_intervals(attacks, to_hit, to_wound, damage, save):
    """
    Returns probabilities for damage intervals.
    """
    n_int = 3
    dmg_lst = damage_list(attacks, damage)
    prob_dmg = prob_damage_saves(attacks, to_hit, to_wound, damage, save)
    exp_dmg = expec_damage_saves(attacks, to_hit, to_wound, damage, save)
    exp_dmg_rnd = int(np.round(exp_dmg))
    exp_dmg_ind = dmg_lst.index(exp_dmg_rnd)
    dmg_prob_int = [["",0]]*n_int
    for i in range(n_int):
        if i == 0:
            label="%d" % exp_dmg_rnd
        else:
            #label="%d +/- %d" %(exp_dmg_rnd, i*damage)
            label="%d-%d" %(exp_dmg_rnd-i*damage, exp_dmg_rnd+i*damage)
        dmg_prob_int[i] = [label, np.sum(prob_dmg[max(exp_dmg_ind-i,0):min(exp_dmg_ind+i+1,(attacks+1))])]
    return dmg_prob_int

def prob_ranges_damage_saves(attacks, to_hit, to_wound, damage, save):
    """
    Returns probabilities for damage intervals.
    """
    n_int = 3
    dmg_lst = damage_list(attacks, damage)
    prob_dmg = prob_damage_saves(attacks, to_hit, to_wound, damage, save)
    exp_dmg = expec_damage_saves(attacks, to_hit, to_wound, damage, save)
    exp_dmg_rnd = int(np.round(exp_dmg))
    exp_dmg_ind = dmg_lst.index(exp_dmg_rnd)
    int_labels = ["exp:", "< 68%", "< 95%"]
    thresholds = [0, 0.68, 0.95]
    dmg_prob_int = [["",0]]*n_int
    prob_range_label = ""
    prob_range = []
    prob_sum = 0
    direction = np.sign(exp_dmg-exp_dmg_rnd)
    if direction == 0:
        direction = np.sign(prob_dmg[exp_dmg_ind+1]-prob_dmg[exp_dmg_ind-1])
    print direction
    k = 0
    l = 0
    for i in range(n_int):
        if i == 0:
            label="%s %d%s" % (int_labels[i], np.round(100*prob_dmg[exp_dmg_ind]), "%")
            prob_range_label = "%d" % exp_dmg_rnd
            prob_range.append(exp_dmg_rnd)
            prob_sum += prob_dmg[exp_dmg_ind]
            k += 1
            print prob_sum
        else:
            label=int_labels[i]
            while prob_sum < thresholds[i]:
                next_ind_p = int(exp_dmg_ind+k*direction)
                if next_ind_p >= 0 and next_ind_p < len(dmg_lst):
                    prob_range.append(damage*next_ind_p)
                    prob_sum+=prob_dmg[next_ind_p]
                    print next_ind_p,prob_sum
                if prob_sum > thresholds[i]:
                    l = 1
                    continue
                next_ind_m = int(exp_dmg_ind-k*direction)
                if next_ind_m >= 0 and next_ind_m < len(dmg_lst):
                    prob_range.append(damage*next_ind_m)
                    prob_sum+=prob_dmg[next_ind_m]
                    print next_ind_m,prob_sum
                k += 1
                #prob_sum = 1
            print prob_range,prob_sum
            #prob_range_label = "%d-%d" % (np.sort(prob_range[-2:])[0],np.sort(prob_range[-2:])[1])
            prob_range_label = "%d-%d" % (np.sort(prob_range[-3:-1])[0],np.sort(prob_range[-3:-1])[1])
            if l == 1:
                next_ind_m = int(exp_dmg_ind-k*direction)
                if next_ind_m >= 0 and next_ind_m < len(dmg_lst):
                    prob_range.append(damage*next_ind_m)
                    prob_sum+=prob_dmg[next_ind_m]
                    print next_ind_m,prob_sum
                k += 1
                l = 0
                
            
        dmg_prob_int[i] = [label, prob_range_label]

    return dmg_prob_int


char = [2,5,4,6] # Charracteristics: Wounds, Move["], Save[+], Bravery
wpn_stats = [1,2,4,3,0,1] # Range["], Attacks, To Hit[+], To Wound[+], Rend, Damage

attacks = 11
to_hit = 4
to_wound = 3
damage = 1
save = 4

reroll_hit_1 = False
reroll_hit_2 = False
reroll_hit_f = False

reroll_wound_1 = False
reroll_wound_2 = False
reroll_wound_f = False

reroll_saves_1 = False
reroll_saves_2 = False
reroll_saves_f = False

print prob_hits(attacks, to_hit)
print expec_hits(attacks, to_hit)

print prob_wounds(attacks, to_hit, to_wound)
print expec_wounds(attacks, to_hit, to_wound)

print damage_list(attacks, damage)
print expec_damage(attacks, to_hit, to_wound, damage)

print prob_damage_saves(attacks, to_hit, to_wound, damage, save)
print expec_damage_saves(attacks, to_hit, to_wound, damage, save)

print prob_damage_saves_intervals(attacks, to_hit, to_wound, damage, save)
print prob_ranges_damage_saves(attacks, to_hit, to_wound, damage, save)


dmg_lst = damage_list(attacks, damage)
p_dmg_svs = prob_damage_saves(attacks, to_hit, to_wound, damage, save)

#plt.bar(dmg_lst, p_dmg_svs, width=damage*0.8, align='center')
#plt.xlim([dmg_lst[0]-0.5*damage,dmg_lst[-1]+0.5*damage])
#plt.xticks(dmg_lst)
#plt.show()

#dmg_lst = damage_list(attacks, damage)
#dmg_prob =  prob_damage_saves(attacks, to_hit, to_wound, damage, save)
#dmg_exp_rnd = int(np.round(expec_damage_saves(attacks, to_hit, to_wound, damage, save)))
#dmg_exp_ind = dmg_lst.index(dmg_exp_rnd)

#print dmg_lst
#print dmg_exp_rnd,":",dmg_prob[dmg_exp_ind]
#print dmg_exp_rnd,"+/-",(damage*1),":",np.sum(dmg_prob[max(dmg_exp_ind-1,0):min(dmg_exp_ind+2,(attacks+1))])
#print dmg_exp_rnd,"+/-",(damage*2),":",np.sum(dmg_prob[max(dmg_exp_ind-2,0):min(dmg_exp_ind+3,(attacks+1))])
#print dmg_exp_rnd,"+/-",(damage*3),":",np.sum(dmg_prob[max(dmg_exp_ind-3,0):min(dmg_exp_ind+4,(attacks+1))])

#plt.plot(dmg_lst,dmg_prob)
#plt.show()
